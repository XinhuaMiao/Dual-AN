


import time
import torch
import torch.nn as nn
import numpy as np

def main_freq_part(x, k, rfft=True):
    # freq normalization
    # start = time.time()
    if rfft:
        xf = torch.fft.rfft(x, dim=1)
    else:
        xf = torch.fft.fft(x, dim=1)
        
    k_values = torch.topk(xf.abs(), k, dim = 1)  
    indices = k_values.indices

    mask = torch.zeros_like(xf)
    mask.scatter_(1, indices, 1)
    xf_filtered = xf * mask

    if rfft:
        x_filtered = torch.fft.irfft(xf_filtered, dim=1).real.float()
    else:
        x_filtered = torch.fft.ifft(xf_filtered, dim=1).real.float()

    
    norm_input = x - x_filtered
    # print(f"decompose take:{ time.time() - start} s")
    return norm_input, x_filtered



class DualAN(nn.Module):
    """FAN first substract bottom k frequecy component from the original series
      

    Args:
        nn (_type_): _description_
    """
    def __init__(self,  seq_len, pred_len, enc_in, freq_topk = 20, rfft=True, **kwargs):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in 
        self.epsilon = 1e-8
        self.freq_topk = freq_topk
        print("freq_topk : ", self.freq_topk )
        self.rfft = rfft
        
        self._build_model()
        self.weight = nn.Parameter(torch.ones(2, self.enc_in))

        self.optimal_window = None
        self.window_sizes = [12, 24, 48]
        self.best_window_size = None
        self.window_scores = {}
        self.means = None
        self.stds = None
        self.pred_means = None
        self.pred_stds = None

    def _build_model(self):
        self.model_freq = MLPfreq(seq_len=self.seq_len, pred_len=self.pred_len, enc_in=self.enc_in)
        self.model_pred = MLPpred(seq_len=self.seq_len, pred_len=self.pred_len, enc_in=self.enc_in)

    def loss(self, true):

        B , O, N= true.shape
        residual, pred_main  = main_freq_part(true, self.freq_topk, self.rfft)
        

        lf = nn.functional.mse_loss

        B, S, E = true.shape
        pad_size = self.optimal_window // 2
        true_padded = torch.nn.functional.pad(true, (0, 0, pad_size, pad_size), mode='replicate')
        
        means_true = torch.zeros_like(true)
        stds_true = torch.zeros_like(true)
        
        for i in range(S):
            start_idx = i
            end_idx = i + self.optimal_window
            window_data = true_padded[:, start_idx:end_idx, :]
            
            mean = window_data.mean(1, keepdim=True)
            std = torch.sqrt(torch.var(window_data, dim=1, keepdim=True, unbiased=False) + 1e-5)
            
            means_true[:, i:i+1, :] = mean
            stds_true[:, i:i+1, :] = std

        all_true = torch.concat([means_true, stds_true], dim=-1)
        all_pred = torch.concat([self.pred_means, self.pred_stds], dim=-1)

        return  lf(self.pred_main_freq_signal, pred_main) + lf(residual, self.pred_residual)
        
    def estimate_optimal_window(self,x):

        try:
            B,S,E = x.shape

            scores = []
            for window in self.window_sizes:
                if window > S:
                    continue
                
                pad_size = window // 2
                x_padded = torch.nn.functional.pad(x, (0, 0, pad_size, pad_size), mode = 'replicate')

                local_std = []
                for i in range(S):
                    window_data = x_padded[:, i:i+window, :]
                    std = torch.std(window_data, dim=1)
                    local_std.append(std)
                
                local_std = torch.stack(local_std, dim=1)

                score = torch.mean(torch.std(local_std, dim=1))
                scores.append(score.item())
                self.window_scores[window] = score.item()
            
            if scores:
                best_idx = np.argmin(scores)
                self.best_window_size = self.window_sizes[best_idx]
            
            return self.best_window_size
        
        except Exception as e:
            print(f"Window estimation failed: {str(e)}")
            return 24

    def adaptive_normalize(self, x, window_size):
        B, S, E = x.shape
        x_normalized = torch.zeros_like(x)
        pad_size = window_size // 2
        x_padded = torch.nn.functional.pad(x, (0, 0, pad_size, pad_size), mode='replicate')
        
        self.means = torch.zeros_like(x)
        self.stds = torch.zeros_like(x)
        
        for i in range(S):
            start_idx = i
            end_idx = i + window_size
            window_data = x_padded[:, start_idx:end_idx, :]
            
            mean = window_data.mean(1, keepdim=True)
            std = torch.sqrt(torch.var(window_data, dim=1, keepdim=True, unbiased=False) + 1e-5)
            
            x_normalized[:, i:i+1, :] = (x[:, i:i+1, :] - mean) / std

            self.means[:, i:i+1, :] = mean
            self.stds[:, i:i+1, :] = std
        
        return x_normalized

    def adaptive_denormalize(self, dec_out):

        denorm_out = dec_out * self.pred_stds + self.pred_means

        return denorm_out

    def normalize(self, input):
        # (B, T, N)
        bs, len, dim = input.shape

        norm_input, x_filtered = main_freq_part(input, self.freq_topk, self.rfft)
        self.pred_main_freq_signal = self.model_freq(x_filtered.transpose(1,2), input.transpose(1,2)).transpose(1,2)
        
        # Sliding Window Adaptive Normalization (SWAN)
        norm_input = self.adaptive_normalize(norm_input, self.optimal_window)

        # Statistical Prediction Module (SPM)
        self.pred_means = self.model_pred(self.means.transpose(1,2), input.transpose(1,2)).transpose(1,2)

        self.pred_stds = self.model_pred(self.stds.transpose(1,2), input.transpose(1,2)).transpose(1,2)

        return norm_input.reshape(bs, len, dim)


    def denormalize(self, output):
        # input:  (B, O, N)
        # station_pred: outputs of normalize
        bs, len, dim = output.shape

        # De-normalization
        output = self.adaptive_denormalize(output)

        # freq denormalize
        self.pred_residual = output
        output = self.pred_residual + self.pred_main_freq_signal
        
        return output.reshape(bs, len, dim)
    
    def forward(self, batch_x, mode='n'):

        if self.optimal_window is None:
            self.optimal_window = self.estimate_optimal_window(batch_x)
            print(f"Estimated optimal window size: {self.optimal_window}")

        if mode == 'n':
            return self.normalize(batch_x)
        elif mode =='d':
            return self.denormalize(batch_x)


class MLPfreq(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in):
        super(MLPfreq, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in
        
        self.model_freq = nn.Sequential(
            nn.Linear(self.seq_len, 64),
            nn.ReLU(),
        )
        
        self.model_all = nn.Sequential(
            nn.Linear(64 + seq_len, 128),
            nn.ReLU(),
            nn.Linear(128, pred_len)
        )


    def forward(self, main_freq, x):
        inp = torch.concat([self.model_freq(main_freq), x], dim=-1)
        return self.model_all(inp)
        
class MLPpred(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in):
        super(MLPpred, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in
        
        self.model_pred = nn.Sequential(
            nn.Linear(self.seq_len, 256),
            nn.ReLU(),
        )
        
        self.model_all = nn.Sequential(
            nn.Linear(256 + seq_len, 512),
            nn.ReLU(),
            nn.Linear(512, pred_len)
        )


    def forward(self, statistics, x):
        inp = torch.concat([self.model_pred(statistics), x], dim=-1)
        return self.model_all(inp)

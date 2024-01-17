import torch
import random
import torch.nn as nn


# class LSTM(nn.Module):
#     def __init__(self, config):
#         super(LSTM, self).__init__()
#         self.input_size = config.model.input_size
#         self.hidden_size = config.model.hidden_size
#         self.output_size = config.model.output_size
#         self.out_seq_length = config.output_length
#
#         self.lstm = nn.LSTM(self.input_size, self.hidden_size, batch_first=True)
#         self.fc = nn.Linear(self.hidden_size, self.output_size)
#
#     def forward(self, x):
#         # x 的形状应该是 (batch_size, sequence_length, input_size)
#         batch_size, in_seq_length, feature_num = x.size()
#
#         # 初始化输出序列
#         output_sequence = torch.zeros(batch_size, self.out_seq_length, feature_num).to(x.device)
#
#         # 初始的 LSTM 隐藏状态和单元状态
#         hidden_state = torch.zeros(batch_size, self.hidden_size).unsqueeze(0).to(x.device)
#         cell_state = torch.zeros(batch_size, self.hidden_size).unsqueeze(0).to(x.device)
#
#         # 逐步生成输出序列
#         for i in range(self.out_seq_length):
#             # 将输入和前一步的输出拼接作为新的输入
#             lstm_input = torch.cat([x[:, i:i + in_seq_length, :], output_sequence[:, :i, :]], dim=1)
#
#             # 将当前输入和隐藏状态传入LSTM
#             lstm_out, (hidden_state, cell_state) = self.lstm(lstm_input, (hidden_state, cell_state))
#
#             # 获取当前步骤的输出
#             step_output = self.fc(lstm_out[:, -1, :].unsqueeze(1))
#
#             # 将当前步骤的输出添加到输出序列中
#             output_sequence[:, i, :] = step_output.squeeze(1)
#
#         return output_sequence

class LSTM(nn.Module):
    def __init__(self, config):
        super(LSTM, self).__init__()
        self.encoder_lstm = nn.LSTM(config.model.input_size, config.model.hidden_size, batch_first=True)
        self.decoder_lstm = nn.LSTM(config.model.hidden_size, config.model.hidden_size, batch_first=True)
        self.out_fc = nn.Linear(config.model.hidden_size, config.model.output_size)
        self.out_seq_length = config.output_length

    def forward(self, x):
        batch_size = x.size(0)
        encoder_outputs, (encoder_hidden, encoder_cell) = self.encoder_lstm(x)
        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell

        decoder_input = encoder_outputs
        decoded_output = []

        for t in range(self.out_seq_length):
            decoder_output, (decoder_hidden, decoder_cell) = self.decoder_lstm(decoder_input, (decoder_hidden, decoder_cell))

            # 计算输出并存储
            output = self.out_fc(decoder_output[:, 0, :].unsqueeze(1))
            decoded_output.append(output)
            decoder_input = torch.cat((decoder_input[:, 1:, :], decoder_output[:, 0, :].unsqueeze(1)), dim=1)

        decoded_output = torch.cat(decoded_output, dim=1)
        return decoded_output

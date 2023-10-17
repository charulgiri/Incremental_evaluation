import pandas as pd
import numpy as np

go_data = pd.read_csv("/Users/charug18/Drive/Work/PhD/Projects/Go Winner Prediction/Go_binary_data_9x9.csv", delimiter=",")
data = go_data.iloc[:,0]
labels = go_data.iloc[:,1]
go_board = []
go_string = []
for x in data:
    black = [1 if bit == "1" else 0 for bit in x]
    white = [1 if bit == "2" else 0 for bit in x]
    black_array = np.reshape(black, (9, 9))
    white_array = np.reshape(white, (9, 9))
    go_board.append( np.stack((black_array, white_array)))
    go_string.append( black+white)
Y=[int(f) for f in labels]
    # print(f"x:{x}\nblack{sum(black)}\nwhite:{sum(white)}")
    # print(go_board, len(go_string))
    # break
print(go_board[0], Y[0])
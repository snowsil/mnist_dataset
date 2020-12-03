import torch
import torch.nn as nn
from deform_conv import * 
def bilinear(x,channel,h_col,w_col,input_h,input_w):
    h_low = int(h_col)
    w_low = int(w_col)
    h_high = h_low + 1
    w_high = w_low + 1
    lw = w_col - w_low
    lh = h_col - h_low
    hw = 1 - lw
    hh = 1 - lh
    v1=0
    v2=0
    v3=0
    v4=0
    if (h_low >=0 and w_low>=0):
        v1 = x[0][channel][h_low][w_low]
    if (h_low >=0 and w_high<=input_w-1):
        v2 = x[0][channel][h_low][w_high]
    if (h_high <= input_h - 1 and w_low >= 0):
        v3 = x[0][channel][h_high][w_low]
    if (h_high <= input_h - 1 and w_high<=input_w-1):
        v4 = x[0][channel][h_high][w_high]
    
    w1 = hh * hw
    w2 = hh * lw
    w3 = lh * hw
    w4 = lh * lw
    val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
    #print(val)
    return val

def my_con2d(input,weight,channel_in,channel_out,kernel_size,dilation,padding,stride,bias,with_bias):
    out_h = (input.shape[2]+2*padding-(dilation*(kernel_size-1)+1))//stride+1
    out_w = (input.shape[3]+2*padding-(dilation*(kernel_size-1)+1))//stride+1
    out = torch.Tensor(1,channel_out,out_h,out_w)
    if padding:
        input_withpad = torch.zeros(1,channel_in,input.shape[2]+2*padding,input.shape[3]+2*padding)
        for c in range(0,channel_in):
            for h in range(0,input.shape[2]):
                for w in range(0,input.shape[3]):
                    input_withpad[0][c][h+padding][w+padding]=input[0][c][h][w]
    else:
        input_withpad = input
    #print(input_withpad)

    for c in range(0,channel_out):
        for h in range(0,out_h):
            for w in range(0,out_w):
                out[0][c][h][w]=0
                for ch_in in range(0,channel_in):
                    h0 = h*stride
                    w0 = w*stride
                    for i in range(0,kernel_size):
                        for j in range(0,kernel_size):
                            out[0][c][h][w] = out[0][c][h][w] + weight[c][ch_in][i][j]*input_withpad[0][ch_in][h0+i*dilation][w0+j*dilation]
                            #print(out[0][c][h][w])
                if with_bias:
                    out[0][c][h][w]=out[0][c][h][w]+bias[c]

    
    return out

def my_deform_conv(x,
            offset,
            mask,
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
            deformable_groups):
    #print("x.shape=",x.shape)
    #print("offset.shape=",offset.shape)
    #print("mask.shape=",mask.shape)
    #print("weight.shape=",weight.shape)
    in_channel=weight.shape[1]
    out_channel=weight.shape[0]
    kernel_size=weight.shape[2]
    input_h=x.shape[2]
    input_w=x.shape[3]
    out_h = (input_h+2*padding-(dilation*(kernel_size-1)+1))//stride+1
    out_w = (input_w+2*padding-(dilation*(kernel_size-1)+1))//stride+1
    #print("in_channel=",in_channel,"out_h=",out_h)
    out = torch.Tensor(1,out_channel,out_h,out_w)
    
    if padding:
      input_withpad = torch.zeros(1,in_channel,x.shape[2]+2*padding,x.shape[3]+2*padding)
      for c in range(0,channel_in):
          for h in range(0,x.shape[2]):
              for w in range(0,x.shape[3]):
                  input_withpad[0][c][h+padding][w+padding]=x[0][c][h][w]
    else:
      input_withpad =x
    column = torch.zeros(1,in_channel,out_h*kernel_size,out_w*kernel_size)
    #print(column.shape)
    #for i in range(0,inchannel)
    for ch in range(0,in_channel):
        for h in range(0,out_h):
            for w in range(0,out_w):
              for i in range(0,kernel_size):
                for j in range(0,kernel_size):
                  h_off=offset[0][2*(i*kernel_size+j)][h][w]
                  w_off=offset[0][2*(i*kernel_size+j)+1][h][w]
                  h_in=h*stride+dilation*i
                  w_in=w*stride+dilation*j
                  h_col = h_in + h_off
                  w_col = w_in + w_off
                  if (h_col>-1 and w_col>-1 and w_col<input_withpad.shape[3] and h_col<input_withpad.shape[2]):
                      column[0][ch][h*kernel_size+i][w*kernel_size+j]= bilinear(input_withpad,ch,h_col,w_col,input_withpad.shape[2],input_withpad.shape[3])
                      #print("column=",column[0][ch][h*kernel_size+i][w*kernel_size+j])
                      column[0][ch][h*kernel_size+i][w*kernel_size+j]=column[0][ch][h*kernel_size+i][w*kernel_size+j] * mask[0][i*kernel_size+j][h][w]
                      #print("mask=",mask[0][i*kernel_size+j][h][w],"column=",column[0][ch][h*kernel_size+i][w*kernel_size+j])
    out = my_con2d(column,weight,in_channel,out_channel,kernel_size,1,0,kernel_size,bias,True)

    return out

if __name__ =="__main__":
    channel_in=1
    channel_out =1
    kernel_size=3
    weight = nn.Parameter(torch.Tensor(1,1,3,3))
    nn.init.kaiming_uniform_(weight, nonlinearity="relu")
    #x=torch.ones(1,1,8,8)
    x=torch.tensor([[[[1,2,3,4,6,3,2,5],
    [5,6,7,8,3,4,6,3],
    [2,3,4,5,3,4,6,3],
    [8,7,5,3,4,6,3,4],
    [2,3,4,5,7,5,3,4],
    [1,2,7,4,5,7,5,3],
    [1,2,7,4,2,56,7,4],
    [9,9,0,5,7,4,2,5]]]]).float()
    #offset = torch.Tensor(1,18,8,8)
    offset=torch.Tensor(1,18,8,8)
    """mask =torch.tensor([[[[-1.0,2.0,3.0],[2.0,3.0,1.0],[2.0,3.0,1.0]],
    [[1.0,5.0,-1.0],[2.0,2.0,2.0],[1.0,-1.0,2.0]]]])
    mask = torch.sigmoid(mask)"""
    mask=torch.Tensor(1,9,8,8)
    mask = torch.sigmoid(mask)
    #mask = torch.ones(1,9,8,8)
    bias=torch.tensor([0.2])
    #out1 = my_con2d(x,weight,1,1,2,1,1,1,bias,True)
    #print(out1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    weight=weight.to(device)
    bias=bias.to(device)
    offset=offset.to(device)
    mask=mask.to(device)
    stride=1
    padding=1
    dilation=1
    groups=1
    deformable_groups=1
    out=my_deform_conv(x,
                offset,
                mask,
                weight,
                bias,
                stride,
                padding,
                dilation,
                groups,
                deformable_groups,
        )
    out1= modulated_deform_conv(
                x,
                offset,
                mask,
                weight,
                bias,
                stride,
                padding,
                dilation,
                groups,
                deformable_groups,
            )
    print("out=",out.float())
    print("out1=",out1)
    print(out.float().to(device)-out1)

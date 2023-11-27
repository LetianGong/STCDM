from . import layers
import torch.nn as nn
import torch

get_act = layers.get_act
default_initializer = layers.default_init


class ST_aware(nn.Module):
  def __init__(self,embed_dim, dim_k, dim_v, cond_out,cond_in, atten_out,atten_in,res_time = 1):
    super(ST_aware, self).__init__()

    self.W_k = nn.Parameter(torch.rand(2,dim_k))
    self.W_q = nn.Parameter(torch.rand(2,dim_k))
    self.W_v = nn.Parameter(torch.rand(2,dim_v))
    self.time_embedding = nn.Embedding(num_embeddings=1000,embedding_dim=embed_dim)
    self.linear1 = nn.Linear(embed_dim,atten_in)
    self.dim_k = torch.tensor(dim_k)
    self.cond_layer = nn.Linear(cond_in,cond_out)
    self.atten_layer = nn.Linear(atten_in*dim_v,atten_out)
    self.relu = nn.ReLU()
    self.norm1 = nn.LayerNorm([res_time,atten_out])
    self.norm2 = nn.LayerNorm([res_time,atten_out])


  def forward(self,cond,x,time_cond,res = False,res_cond = None,res_x= None,res_time = 1):
    K = torch.cat((cond,x),dim=-1) @ self.W_k# batch_size*1*317*dim_k
    Q = torch.cat((cond,x),dim=-1) @ self.W_q# batch_size*1*317*dim_k
    t_emb = torch.reshape(self.linear1(self.time_embedding(time_cond)),(x.shape[0],1,x.shape[2],1))
    V = torch.cat((t_emb.repeat(1,res_time,1,1),x),dim=-1) @ self.W_v# batch_size*1*317*dim_v
    atten = nn.Softmax(dim=-1)((Q @ K.permute(0,1,3,2))) * (1/torch.sqrt(self.dim_k))# batch_size*1*317*317
    atten = atten @ V# batch_size*1*317*dim_v
    atten = torch.reshape(atten,(x.shape[0],x.shape[1],-1))
    cond = torch.reshape(cond,(x.shape[0],x.shape[1],-1))
    cond_out = self.relu(self.cond_layer(cond))# batch_size*1*128*1
    x_out = self.relu(self.atten_layer(atten))# batch_size*1*128*1
    cond_out = self.norm1(cond_out).reshape(*cond_out.shape,1)
    x_out = self.norm2(x_out).reshape(*x_out.shape,1)
    if res:
      cond_out = torch.cat((cond_out,res_cond),dim=-3)
      x_out = torch.cat((x_out,res_x),dim=-3)
    return cond_out,x_out
  

class ST_Unet(nn.Module):
  def __init__(self, FLAGS):
    super().__init__()
    self.POI_num = FLAGS.POI_num
    self.ST_down1 = ST_aware(embed_dim=FLAGS.embed_dim,dim_k = 256,dim_v=256,cond_out=128,cond_in=FLAGS.POI_num,atten_out=128,atten_in=FLAGS.POI_num)
    self.ST_down2 = ST_aware(embed_dim=FLAGS.embed_dim,dim_k = 256,dim_v=256,cond_out=32,cond_in=128,atten_out=32,atten_in=128)
    self.ST_down3 = ST_aware(embed_dim=FLAGS.embed_dim,dim_k = 256,dim_v=256,cond_out=32,cond_in=32,atten_out=32,atten_in=32)
    self.ST_up1 = ST_aware(embed_dim=FLAGS.embed_dim,dim_k = 256,dim_v=256,cond_out=128,cond_in=32,atten_out=128,atten_in=32,res_time = 2)
    self.ST_up2 = ST_aware(embed_dim=FLAGS.embed_dim,dim_k = 256,dim_v=256,cond_out=FLAGS.POI_num,cond_in=128,atten_out=FLAGS.POI_num,atten_in=128,res_time = 4)
    self.out_layer = nn.Linear(in_features=4*FLAGS.POI_num,out_features=FLAGS.POI_num)
    self.act = nn.Tanh()

    
  def forward(self, x, time_cond, cond):
    x = torch.reshape(x,(x.shape[0],1,self.POI_num,1))
    cond = torch.reshape(cond,(cond.shape[0],1,self.POI_num,1))
    cond1,x1 = self.ST_down1(cond,x,time_cond)
    cond2,x2 = self.ST_down2(cond1,x1,time_cond)
    cond3,x3 = self.ST_down3(cond2,x2,time_cond,res = True,res_cond = cond2,res_x= x2)
    cond4,x4 = self.ST_up1(cond3,x3,time_cond,res = True,res_cond = cond1.repeat(1,2,1,1),res_x= x1.repeat(1,2,1,1),res_time = 2)
    cond5,x5 = self.ST_up2(cond4,x4,time_cond,res_time=4)
    outputs = self.act(self.out_layer(x5.reshape(x.shape[0],-1)))

    return outputs
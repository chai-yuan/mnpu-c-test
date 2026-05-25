"""Compare Python manual inference vs C inference on test images."""
import os, glob, subprocess, math, numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import warnings; warnings.filterwarnings('ignore')

mp = 'out/mnist_full_int8.tflite'
i = tf.lite.Interpreter(model_path=mp); i.allocate_tensors()
d = i.get_tensor_details()
ops = i._get_ops_details()
fc = [op for op in ops if op['op_name'] == 'FULLY_CONNECTED']

def gqp(idx):
    dq = d[idx]; qp = dq.get('quantization_parameters', None)
    if qp is not None and qp['scales'].size > 0:
        return np.array(qp['scales'],dtype=np.float32), np.array(qp['zero_points'],dtype=np.int32)
    s,z = dq['quantization']; return np.array(s,dtype=np.float32), np.array(z,dtype=np.int32)

def qm(dm):
    if dm==0.0: return np.int32(0), np.int32(0)
    m,s = math.frexp(dm)
    q=int(round(m*(1<<31)))
    if q==(1<<31): q//=2; s+=1
    if s<-31: s=0; q=0
    return np.int32(q), np.int32(s)

def srh(a,b):
    ab=np.int64(a)*np.int64(b)
    n=np.where(ab>=0, np.int64(1<<30), np.int64(1-(1<<30)))
    r=((ab+n)>>31).astype(np.int32)
    return np.where((a==np.int32(-2147483648))&(b==np.int32(-2147483648)), np.int32(2147483647), r)

def rdp(x,e):
    if e==0: return x
    m=(np.int32(1)<<np.int32(e))-np.int32(1)
    r=x&m; t=(m>>1)+np.where(x<0,np.int32(1),np.int32(0))
    return (x>>np.int32(e))+np.where(r>t,np.int32(1),np.int32(0))

def mqm(x,qm,sh):
    L=np.where(sh>0,sh,np.int32(0)); R=np.where(sh>0,np.int32(0),-sh)
    return rdp(srh(np.left_shift(x,L), qm), R)

def extract_layer(fc_op):
    ii,wi,bi = fc_op['inputs']; oi=fc_op['outputs'][0]
    _,iz=gqp(ii); ws,wz=gqp(wi)
    w=i.get_tensor(wi); b=i.get_tensor(bi)
    os_,oz=gqp(oi)
    e=(float(np.squeeze(_)) * ws.astype(np.float64))/float(np.squeeze(os_))
    qms=np.zeros(ws.size,dtype=np.int32); shs=np.zeros(ws.size,dtype=np.int32)
    for j in range(ws.size): qms[j],shs[j]=qm(e[j])
    return w,b,qms,shs,int(np.squeeze(iz)),int(np.squeeze(oz))

def dense_int8(x,w,b,qms,shs,zi,zo):
    x32=x.flatten().astype(np.int32)-zi
    acc=np.dot(x32.reshape(1,-1),w.astype(np.int32).T)+b
    out=np.zeros((1,acc.shape[1]),dtype=np.int32)
    for c in range(acc.shape[1]):
        out[0,c]=mqm(np.int32(acc[0,c]), qms[c], shs[c])
    return np.clip(out+zo, -128, 127).astype(np.int8)

w1,b1,qm1,sh1,_,oz1=extract_layer(fc[0])
w2,b2,qm2,sh2,_,oz2=extract_layer(fc[1])
_,inz=gqp(fc[0]['inputs'][0]); inz=int(np.squeeze(inz))

files=sorted(glob.glob('out/test_img_*.bin'))
print(f'{"="*60}')
print(f'{"file":>30s}  lbl  py  clang  match?')
print(f'{"-"*60}')

same=0; tot=0
for f in files:
    label=int(os.path.basename(f).split('_label_')[1].split('.')[0])
    with open(f,'rb') as fh: dat=fh.read()
    img=np.frombuffer(dat,dtype=np.int8).reshape(1,784)
    l1o=dense_int8(img,w1,b1,qm1,sh1,inz,oz1)
    l2o=dense_int8(l1o,w2,b2,qm2,sh2,oz1,oz2)
    pp=np.argmax(l2o[0])
    cp=subprocess.run(['./build/native/mnist-int8','out/model.bin',f],
                       capture_output=True,text=True).stdout.strip()
    m='Y' if str(pp)==cp else 'N'
    tot+=1
    if str(pp)==cp: same+=1
    print(f'{os.path.basename(f):>30s}  {label:>3d}  {pp:>3d}  {cp:>4s}  {m}')

print(f'{"-"*60}')
print(f'Agreement: {same}/{tot}')

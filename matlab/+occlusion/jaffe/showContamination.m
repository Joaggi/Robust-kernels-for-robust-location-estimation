addpath('../../Datasets')
load('Jaffe_occlusion_contamination')
dim=[32,32]

subplot(6,5,1), subimage(reshape(data_real(1,:)/256,dim(1),dim(2)))
subplot(6,5,6), subimage(reshape(data_real(4,:)/256,dim(1),dim(2)))
subplot(6,5,11), subimage(reshape(data_real(15,:)/256,dim(1),dim(2)))
subplot(6,5,16), subimage(reshape(data_real(50,:)/256,dim(1),dim(2)))
subplot(6,5,21), subimage(reshape(data_real(100,:)/256,dim(1),dim(2)))
subplot(6,5,26), subimage(reshape(data_real(120,:)/256,dim(1),dim(2)))

subplot(6,5,4), subimage(reshape(data_real(59,:)/256,dim(1),dim(2)))
subplot(6,5,9), subimage(reshape(data_real(60,:)/256,dim(1),dim(2)))
subplot(6,5,14), subimage(reshape(data_real(65,:)/256,dim(1),dim(2)))
subplot(6,5,19), subimage(reshape(data_real(70,:)/256,dim(1),dim(2)))
subplot(6,5,24), subimage(reshape(data_real(200,:)/256,dim(1),dim(2)))
subplot(6,5,29), subimage(reshape(data_real(213,:)/256,dim(1),dim(2)))


load('ATT')
subplot(6,5,2), subimage(reshape(data_contamination(1,:)/256,dim(1),dim(2)))
subplot(6,5,7), subimage(reshape(data_contamination(4,:)/256,dim(1),dim(2)))
subplot(6,5,12), subimage(reshape(data_contamination(15,:)/256,dim(1),dim(2)))
subplot(6,5,17), subimage(reshape(data_contamination(50,:)/256,dim(1),dim(2)))
subplot(6,5,22), subimage(reshape(data_contamination(100,:)/256,dim(1),dim(2)))
subplot(6,5,27), subimage(reshape(data_contamination(120,:)/256,dim(1),dim(2)))


subplot(6,5,5), subimage(reshape(data_contamination(59,:)/256,dim(1),dim(2)))
subplot(6,5,10), subimage(reshape(data_contamination(60,:)/256,dim(1),dim(2)))
subplot(6,5,15), subimage(reshape(data_contamination(65,:)/256,dim(1),dim(2)))
subplot(6,5,20), subimage(reshape(data_contamination(70,:)/256,dim(1),dim(2)))
subplot(6,5,25), subimage(reshape(data_contamination(200,:)/256,dim(1),dim(2)))
subplot(6,5,30), subimage(reshape(data_contamination(213,:)/256,dim(1),dim(2)))
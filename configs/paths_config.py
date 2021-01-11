import os

dataset_paths = {
	'celeba_train': '',
	'celeba_test': '',
	'celeba_train_sketch': '',
	'celeba_test_sketch': '',
	'celeba_train_segmentation': '',
	'celeba_test_segmentation': '',
	'ffhq': '',
}

def get_model_path(name):
	v = model_paths.get(name,None)
	if v is None:
		return v
	prefix = os.environ.get('PRETRAIN_PATH','./')
	return os.path.join(prefix,v)

model_paths = {
	'stylegan_ffhq': 'stylegan2-ffhq-config-f.pt',
	'ir_se50': 'model_ir_se50.pth',
	'circular_face': 'CurricularFace_Backbone.pth',
	'mtcnn_pnet': 'mtcnn/pnet.npy',
	'mtcnn_rnet': 'mtcnn/rnet.npy',
	'mtcnn_onet': 'mtcnn/onet.npy',
	'shape_predictor': 'shape_predictor_68_face_landmarks.dat'
}

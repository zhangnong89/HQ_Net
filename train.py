import config as config
import os
import model.model as model_net
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint


class ParallelModelCheckpoint(ModelCheckpoint):
    def __init__(self, model, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        self.single_model = model
        super(ParallelModelCheckpoint,self).__init__(filepath, monitor, verbose,save_best_only, save_weights_only,mode, period)

    def set_model(self, model):
        super(ParallelModelCheckpoint,self).set_model(self.single_model)


def train():
	cfg = config.Config()
	num_gpus = int(len(cfg.gpus.split(",")))
	X, y = cfg.data_load()
	X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.12, random_state=42)
	cnn_Model = model_net.NN_Model()
	if num_gpus == 1:
		single_model = cnn_Model.model()
		model = single_model
		if os.path.exists(cfg.checkpoints_save_path):
			single_model.load_weights(cfg.checkpoints_save_path)
	else:
		parallel_model, single_model = cnn_Model.model()
		model = parallel_model
		if os.path.exists(cfg.checkpoints_save_path):
			single_model.load_weights(cfg.checkpoints_save_path)
	# compile the model
	model.compile(optimizer=cfg.optimizers(lr=cfg.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), loss=cfg.loss, metrics=['accuracy'])
	
	if num_gpus == 1:
		checkpointer = ModelCheckpoint(filepath=cfg.checkpoints_save_path, verbose=1, save_weights_only=False, save_best_only=True)
	else:
		checkpointer = ParallelModelCheckpoint(single_model, filepath=cfg.checkpoints_save_path, verbose=1, save_weights_only=False, save_best_only=True)
	history = model.fit(X_train, y_train, batch_size=cfg.batch_size, epochs=cfg.epochs,
	                    callbacks=[cfg.early_stopping, checkpointer], shuffle=True, validation_split=0.1)
	
	with open('training.log', 'w') as f:
		f.write(str(history.history) + '\n')
		
	# evaluate the model
	train_loss, train_accuracy = model.evaluate(X_train, y_train, batch_size=32)
	valid_loss, valid_accuracy = model.evaluate(X_valid, y_valid, batch_size=32)
	print('\nLoss: %.2f, Accuracy: %.2f%%' % (train_loss, train_accuracy*100))
	print('\nLoss: %.2f, Accuracy: %.2f%%' % (valid_loss, valid_accuracy*100))
	#model.save_weights(cfg.model_save_path)
	single_model.save(cfg.model_save_path)

	
if __name__ == '__main__':
	train()

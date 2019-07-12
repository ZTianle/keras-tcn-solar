import keras
from tcn import compiled_tcn
from utils import data_generator
from matplotlib import pyplot
import matplotlib.pyplot as plt

x, y = data_generator(seq_length=300)



x_train, y_train = x[:int(len(y)*0.8),:,:] , y[:int(len(y)*0.8),:]
x_test, y_test = x[int(len(y)*0.8):,:,:] , y[int(len(y)*0.8):,:]

#min_max_scaler = preprocessing.MinMaxScaler()
#X_train_minmax = min_max_scaler.fit_transform(X_train)
#X_test = np.array([[ -3., -1., 4.]])
#X_test_minmax = min_max_scaler.transform(X_test)


class PrintSomeValues(keras.callbacks.Callback):

    def on_epoch_begin(self, epoch, logs={}):
        print(f'x_test[0:1] = {x_test[0:1]}.')
        print(f'y_test[0:1] = {y_test[0:1]}.')
        print(f'pred = {self.model.predict(x_test[0:1])}.')


def run_task():
    model = compiled_tcn(return_sequences=False,
                         num_feat=x_train.shape[2],
                         num_classes=0,
                         nb_filters=36,
                         kernel_size=8,
                         dilations=[2 ** i for i in range(9)],
                         nb_stacks=1,
                         max_len=x_train.shape[1],
                         use_skip_connections=True,
                         regression=True,
                         dropout_rate=0)

    print(f'x_train.shape = {x_train.shape}')
    print(f'y_train.shape = {y_train.shape}')

    psv = PrintSomeValues()

    # Using sparse softmax.
    # http://chappers.github.io/web%20micro%20log/2017/01/26/quick-models-in-keras/
    model.summary()

    #history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50,
    #          callbacks=[psv], batch_size=128)

    # 创建一个权重文件保存文件夹logs
    log_dir = "logs/"
    # 记录所有训练过程，每隔一定步数记录最大值
    tensorboard = keras.callbacks.TensorBoard(log_dir=log_dir)
    checkpoint = keras.callbacks.ModelCheckpoint(log_dir + "best_weights.h5",
                                 monitor="val_loss",
                                 mode='min',
                                 save_weights_only=False,
                                 save_best_only=False,
                                 verbose=1,
                                 period=1)

    callback_lists = [tensorboard, checkpoint]

    history = model.fit(x_train, y_train, validation_split=0.1, shuffle=True, epochs=20,
                        callbacks= callback_lists, batch_size=128)

    y_pred = model.predict(x_test)

    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title('model train vs validation loss')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'validation'], loc='upper right')
    pyplot.show()

    x_axix=range(len(y_test))
    plt.plot(x_axix, y_test, color='green', label='test_ture')
    plt.plot(x_axix, y_pred, color='red', label='test_tcn')
    plt.legend()  # 显示图例
    plt.xlabel('times')
    plt.ylabel('solar')
    plt.show()
    test_mse_score, test_mae_score = model.evaluate(x_test, y_test, verbose=1)
    print('Test mse score:', test_mse_score)
    print('Test mae score:', test_mae_score)

    # 保存模型
    model.save('my_model.h5')


if __name__ == '__main__':
    run_task()
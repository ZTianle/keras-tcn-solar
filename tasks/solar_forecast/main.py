import keras
from tcn import compiled_tcn
from utils import data_generator
from matplotlib import pyplot
import matplotlib.pyplot as plt
x, y = data_generator(seq_length=100)
x_train, y_train = x[:int(len(y)*0.8),:,:] , y[:int(len(y)*0.8),:]
x_test, y_test = x[int(len(y)*0.8):,:,:] , y[int(len(y)*0.8):,:]



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
    history = model.fit(x_train, y_train, validation_split=0.1, shuffle=False, epochs=20,
                        callbacks=[keras.callbacks.TensorBoard(log_dir='./tmp/log')], batch_size=128)

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
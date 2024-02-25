from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
from time import strftime


my_dataset_path = 'D://test2'
my_image_size = (32,32)
my_input_shape = my_image_size + (1,)
# 指定训练次数
my_train_epochs = 3
# 指定batch
my_batch = 32
# shuffle buffer size
my_shuffle_buffer_size = 100000

AUTOTUNE = tf.data.experimental.AUTOTUNE

# 获取所有文件路径
dataset_path = pathlib.Path(my_dataset_path)
all_images_paths = [str(path) for path in list(dataset_path.glob('*/*'))]
print('所有文件的路径:', all_images_paths)
print('文件总数:', len(all_images_paths))

# 获取标签名称
label_name = [i.name for i in dataset_path.iterdir() if i.is_dir()]
print('标签名称:', label_name)
# 因为训练时参数必须为数字，因此为标签分配数字索引
label_index = dict((name,index+1)for index,name in enumerate(label_name))
print('为标签分配数字索引:', label_index)

# 将图片与标签的数字索引进行配对(number encodeing)
number_encodeing = [label_index[i.split('\\')[2]]for i in all_images_paths]
print('number_encodeing:', number_encodeing, type(number_encodeing))
label_one_hot = tf.keras.utils.to_categorical(number_encodeing)
print('label_one_hot:', label_one_hot)


def process(path,label):
    # 读入图片文件
    image = tf.io.read_file(path)
    # 将输入的图片解码为gray或者rgb
    image = tf.image.decode_jpeg(image, channels=my_input_shape[2])
    # 调整图片尺寸以满足网络输入层的要求
    image = tf.image.resize(image, my_image_size)
    # 归一化
    image /= 255.
    return image,label

# 将数据与标签拼接到一起
path_ds = tf.data.Dataset.from_tensor_slices((all_images_paths, tf.cast(label_one_hot, tf.int32)))

image_label_ds = path_ds.map(process, num_parallel_calls=AUTOTUNE)
# dataset = image_label_ds.map(lambda x, y: (tf.keras.layers.RandomZoom(height_factor=(0, 0.1),width_factor=(0, 0.1),fill_mode='constant')(x), y))
# dataset = dataset.prefetch(tf.data.AUTOTUNE)
print('image_label_ds:', image_label_ds)
steps_per_epoch=tf.math.ceil(len(all_images_paths)/my_batch).numpy()
print('steps_per_epoch', steps_per_epoch)

# 打乱dataset中的元素并设置batch
image_label_ds = image_label_ds.shuffle(my_shuffle_buffer_size).batch(my_batch)
# 输入层
input_data = tf.keras.layers.Input(shape=my_input_shape)
predeal=tf.keras.layers.RandomZoom(height_factor=(0, 0.1),width_factor=(0, 0.1),fill_mode='constant')(input_data)
predeal=tf.keras.layers.RandomRotation(factor=0.25,fill_mode='constant')(predeal)
preprocess_data=tf.keras.layers.GaussianNoise(stddev=0.1)(predeal)
# 第一层
middle = tf.keras.layers.Conv2D(32, kernel_size=[5,5], strides=(1,1), padding='valid', activation=tf.nn.relu)(preprocess_data)
middle = tf.keras.layers.MaxPool2D(pool_size=[2,2], strides=2)(middle)
# 第二层
middle = tf.keras.layers.Conv2D(64, kernel_size=[5,5], strides=(1,1), padding='valid', activation=tf.nn.relu)(middle)
middle = tf.keras.layers.MaxPool2D(pool_size=[2,2], strides=2)(middle)

# 铺平
dense = tf.keras.layers.Flatten()(middle)
dense = tf.keras.layers.Dropout(0.1)(dense)
dense = tf.keras.layers.Dense(100, activation='relu')(dense)
# 输出层
output_data = tf.keras.layers.Dense(len(label_name)+1, activation='softmax')(dense)
# 确认输入位置和输出位置

preprocess_model=tf.keras.Model(input_data,preprocess_data)
model = tf.keras.Model(inputs=preprocess_data, outputs=output_data)

# 定义模型的梯度下降和损失函数
model.compile(optimizer=tf.optimizers.Adam(1e-4), 
            loss=tf.losses.categorical_crossentropy,
            metrics=['accuracy'])

# 打印模型结构
model.summary()

# 开始训练
start_time = strftime("%Y-%m-%d %H:%M:%S")
preprocessed_ds = image_label_ds.map(
    lambda x, y: (preprocess_model(x), y),
    num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

history = model.fit(
    preprocessed_ds,
    epochs=my_train_epochs,
    verbose=1,
    steps_per_epoch=int(steps_per_epoch))

end_time = strftime("%Y-%m-%d %H:%M:%S")
print('开始训练的时间：', start_time)
print('结束训练的时间：', end_time)


# Convert Keras model to ConcreteFunction
full_model = tf.function(lambda x: model(x))
full_model = full_model.get_concrete_function(
tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

# Get frozen ConcreteFunction
frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()

layers = [op.name for op in frozen_func.graph.get_operations()]
print("-" * 50)
print("Frozen model layers: ")
for layer in layers:
    print(layer)

print("-" * 50)
print("Frozen model inputs: ")
print(frozen_func.inputs)
print("Frozen model outputs: ")
print(frozen_func.outputs)

# Save frozen graph from frozen ConcreteFunction to hard drive
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
		logdir="./frozen_models",
		name="frozen_graph.pb",
		as_text=False)
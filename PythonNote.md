# Python

## numpy

### np.newaxis

**np.newaxis：**对选取的数据增加一个维度。

```python
'''
np.newaxis返回修改值，原值不变。
'''
import numpy as np
data = np.random.randint(1,3,size=(3,2))
a = data[np.newaxis,:]#最外维增加一维
b = data[:,np.newaxis]#选取全部数据进行维度增加
c = data[..., np.newaxis]#最内维增加一维
d = data[0:2,np.newaxis]#选取0和1位置数据进行维度增加
>>>data
array([[1, 1],
       [2, 1],
       [1, 1]])
>>>data.shape
(3, 2)
'(1)[np.newaxis,:]'
>>>a
array([[[1, 1],
        [2, 1],
        [1, 1]]])
>>>a.shape
(1, 3, 2)
'(2)[:,np.newaxis]'
>>>b
array([[[1, 1]],
       [[2, 1]],
       [[1, 1]]])
>>>b.shape
(3, 1, 2)
'(3)[...,np.newaxis]'
>>>c
array([[[1],
        [1]],
       [[2],
        [1]],
       [[1],
        [1]]])
>>>c.shape
(3, 2, 1)
'(4)[0:2,np.newaxis]'
>>>d
array[[[1 2]],[[1 1]]]
>>>d.shape
(2, 1, 2)
```

### np.repeat()

np.repeat()：沿着横纵轴方向重复每一个元素。

```python
>>>np.repeat(3,3)
array([3, 3, 3])

>>>x = np.array([[1,2],[3,4]])
#默认将x展成一维，再对每个元素进行重复
>>>np.repeat(x, 2)
array([1, 1, 2, 2, 3, 3, 4, 4])
'从外到内，维度从0到1'
#沿着纵轴重复，增加行数
>>>np.repeat(x, 2,axis=0)
array([[1, 2],
       [1, 2],
       [3, 4],
       [3, 4]])
#沿着横轴重复，增加列数
>>>np.repeat(x, 2, axis=1)
array([[1, 1, 2, 2],
       [3, 3, 4, 4]])
```

### np.linspace()

**np.linspace()：**构造等差数列，生成指定范围内指定个数的等间距一维数组。

```python
'''
linspace(start,stop,num,endpoint,retstep)
参数：
[1]在[start,stop]内，返回num个等间距的数字，间距为(stop-start)/num。
[2]endpoint=True则stop为最后一个值，默认endpoint=True。
[3]retstep=True则返回样本间的间隙。
'''
#构造等差数列 开始值 结束值 共几个数字
c = np.linspace(1,5,5)
>>[1,2,3,4,5]
>>>c.shape
(5,)
```

### np.logspace()

**np.logspace()：**构造等比数列。

```python
'''
logspace(start,end,num,base,endpoint)
base：幂的底数，默认是10。
start/end：幂的指数。
其他同linspace()
'''
a = np.logspace(0,4,5,base=2)
>>>[1., 2., 4., 8., 16.]
```



### np.arange()

1、numpy arange 和 python range 区别：
		**arange**可以生成浮点类型，而**range**只能是整数类型。
2、np.arange()：在给定的范围[start,stop)内返回均匀间隔step的数组对象ndarray，常用于循环。

```python
#start、stop、step：可正负，可小数
#start默认为0，step默认为1
a = np.arange(3)
>>[0 1 2]
a = np.arange(3,9)
>>[3 4 5 6 7 8]

>>>a = np.arange(3,9,0.1)
array([3. , 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4. , 4.1, 4.2,
       4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5. , 5.1, 5.2, 5.3, 5.4, 5.5,
       5.6, 5.7, 5.8, 5.9, 6. , 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8,
       6.9, 7. , 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8. , 8.1,
       8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9])
```

### np.random.seed()

**随机种子：**随机数的初始值，随机种子相同，得到的随机数一定也相同。

```python
#设置随机种子，使测试结果可复现
'''
设置seed()里的值等价于获得一个容纳随机数的容器，一个数字代表一个容器，当seed()里的值相同，容器即一样，亦即每次拿出的随机数就相同（同一容器按相同的方式随机取数字，同一容器取出的随机数相同）。
seed(num)：num值可随便设置。
seed()只作用一次随机数的生成，再下一次随机数生成不受影响。
'''
import numpy as np
np.random.seed(0)
#a、b不相同，a属于seed(0)容器，b不属于seed(0)
a = np.random.rand(4,3)
b = np.random.rand(4,3)
```

### np.random.rand()

**np.random.rand()：**生成一个ndarray对象，该对象为一个或一组n维服从"0~1"均匀分布的随机样本值，[0,1)。

```python
'''
numpy.random.rand(d1,...dn)
dn：n维具有的样本值个数
注：使用方法与np.random.randn()同
'''
batch_size = 8
>>>np.random.rand(4, batch_size, 1)
array([[[0.84894786],
        [0.56547657],
        [0.68389452],
        [0.43427059],
        [0.84497871],
        [0.72068617],
        [0.11488394],
        [0.00120193]],
       [[0.69214443],
        [0.46332693],
        [0.99618256],
        [0.72556112],
        [0.00724553],
        [0.51118714],
        [0.75763048],
        [0.12899612]],
       [[0.41490381],
        [0.55386294],
        [0.35127252],
        [0.24396263],
        [0.09807242],
        [0.86270158],
        [0.00608012],
        [0.71037031]],
       [[0.68606525],
        [0.55012171],
        [0.19125298],
        [0.40674154],
        [0.75297308],
        [0.74912834],
        [0.21827876],
        [0.09115221]]])

a, b = np.random.rand(2, batch_size, 1)
>>>a
array([[0.0787164 ],
       [0.41913545],
       [0.55698636],
       [0.78815134],
       [0.5722761 ],
       [0.32456097],
       [0.08186641],
       [0.06123344]])
>>>a.shape
(8, 1)
>>>n_steps = 5
>>>c = np.linspace(1, 5, n_steps)
>>>c
array([1., 2., 3., 4., 5.])
>>>c.shape
(5,)
>>>c-a
array([[0.9212836 , 1.9212836 , 2.9212836 , 3.9212836 , 4.9212836 ],
       [0.58086455, 1.58086455, 2.58086455, 3.58086455, 4.58086455],
       [0.44301364, 1.44301364, 2.44301364, 3.44301364, 4.44301364],
       [0.21184866, 1.21184866, 2.21184866, 3.21184866, 4.21184866],
       [0.4277239 , 1.4277239 , 2.4277239 , 3.4277239 , 4.4277239 ],
       [0.67543903, 1.67543903, 2.67543903, 3.67543903, 4.67543903],
       [0.91813359, 1.91813359, 2.91813359, 3.91813359, 4.91813359],
       [0.93876656, 1.93876656, 2.93876656, 3.93876656, 4.93876656]])
>>>(c-a).shape
(8, 5)
>>>((c-a)*c)
array([[ 0.9212836 ,  3.8425672 ,  8.7638508 , 15.6851344 , 24.606418  ],
       [ 0.58086455,  3.16172909,  7.74259364, 14.32345819, 22.90432273],
       [ 0.44301364,  2.88602728,  7.32904093, 13.77205457, 22.21506821],
       [ 0.21184866,  2.42369731,  6.63554597, 12.84739463, 21.05924328],
       [ 0.4277239 ,  2.85544779,  7.28317169, 13.71089558, 22.13861948],
       [ 0.67543903,  3.35087806,  8.02631709, 14.70175612, 23.37719516],
       [ 0.91813359,  3.83626718,  8.75440077, 15.67253436, 24.59066795],
       [ 0.93876656,  3.87753312,  8.81629968, 15.75506624, 24.6938328 ]])
>>>((c-a)*c).shape
(8, 5)
```

### np.random.randn()

**np.random.randn()：**用法和rand()相同，数组元素符合标准正态分布N(0,1)。

```python
'''
numpy.random.randn(d1,...dn)
'''
```

### np.random.randint()

**np.random.randint()：**返回一个由a到b之间随机整数构成size形状的ndarray对象。

```python
'''
np.random.randint(a,b,size=(c,d))
从[a,b)随机取c*d个整数，构成c行d列数组。
'''
>>>np.random.randint(2,4,(2,2))
array([[3,2],[3,3]])
```

### np.random.random()

**np.random.random()：**生成`size`尺寸的随机浮点数数组，浮点数从`[0.0, 1.0)`中随机。

```python
'np.random.random(size=None)'
>>>np.random.random((3, 4))#生成3行4列的浮点数，维度2
array([[0.63555256, 0.72659315, 0.29029868, 0.76210139],
       [0.37999286, 0.6140207 , 0.53935961, 0.55272388],
       [0.81801144, 0.50752384, 0.91947224, 0.14889305]])
>>>np.random.random((2, 3, 4))#生成3行4列的浮点数，维度3
array([[[0.81670827, 0.19777878, 0.91000583, 0.99528314],
        [0.91530841, 0.4945461 , 0.12852176, 0.86353084],
        [0.96499686, 0.77510463, 0.58097892, 0.10637159]],
       [[0.566005  , 0.23129435, 0.52181051, 0.20799262],
        [0.12962714, 0.51261245, 0.514109  , 0.31633069],
        [0.65525944, 0.37226966, 0.81251317, 0.33227821]]])
```

### np.random.random_sample()

**np.random.random_sample()：**与np.random.random用法相同。

```python
'np.random.random_sample(size=None)'
>>>np.random.random_sample()
0.2244731575380413
>>>type(np.random.random_sample())
<class 'float'>
>>>np.random.random_sample((3,))
array([0.12808342, 0.27356875, 0.78981981])
>>>5 * np.random.random_sample((3,2)) - 5
array([[-2.86419098, -3.11387737],
       [-2.77948548, -1.60698822],
       [-4.11583226, -2.44705993]])
```

### np.random.sample()

**np.random.sample()：**与np.random.random用法相同。

```python
'np.random.sample(size=None)'
```



### np.random.choice()

**np.random.choice()：**从数组`ndarray`、列表`list`或元组`tuple`中随机抽取元素，且它们必须是一维。

```python
'''
np.random.choice(a, size=None, replace=True, p=None)
参数：
a：从 a 中随机抽取数字，并组成指定大小（size）的数组。
replace：默认为True，True表示可以取相同的数字，False表示不可以取相同数字。
p：与 a 相对应，表示取 a 中每个元素的概率，默认为选取每个元素的概率相同。
'''
#相当于np.random.randint(0,5)
>>>np.random.choice(5)#从[0,5)中随机输出一个随机数
2

#相当于np.random.randint(0,5,3)
>>>np.random.choice(5,3)#从[0,5)中输出3个数字并组成一维数组ndarray
array([4, 4, 2])

L = [1,2,3,4,5]
T = (2,4,6,2)
A = np.array([4,2,1])
>>>np.random.choice(L,5)
array([5, 4, 2, 5, 5])
>>>np.random.choice(T,5)
array([6, 4, 6, 6, 6])
>>>np.random.choice(A,5)
array([1, 2, 4, 2, 1])

#有 p 时
a = ['A','B','C','D']
>>>np.random.choice(a, 5, p=[0.5,0.1,0.2,0.2])
array(['A', 'A', 'C', 'D', 'A'], dtype='<U1')
```

### np.random.multinomial()

**np.random.multinomial()：**该函数表示根据一个概率数组，取若干次，取到一个次数分布数组。

```python
'''
np.random.multinomial(n, pvals, size=None)
参数：
n：实验次数，int
pvals：概率数组，浮点数序列，长度为p，即p个不同结果的概率，概率和为1
size：输出维度，默认为1，即 1*pvals
'''
#掷骰子20次
>>np.random.multinomial(20,[1/6.]*6, size=1)
array([[5, 3, 5, 2, 1, 4]])
#掷骰子20次，再掷20次
>>np.random.multinomial(20,[1/6.]*6,size=2)
array([[5, 0, 4, 5, 4, 2],
       [1, 4, 4, 2, 4, 5]])
```

### np.random.normal()

**np.random.normal()：**从正态（高斯）分布中抽取随机样本

```python
'''
np.random.normal(loc=0.0, scale=1.0, size=None)
参数：
[1]loc：分布的均值（中心）。为浮点型数据或浮点型数据组成的数组
[2]scale：分布的标准差（宽度）。为浮点型数据或浮点型数据组成的数组
[3]size：输出值的维度。为整数或者整数组成的元组，可选参数。如果给定的维度为(m, n, k)，那么就从分布中抽取m * n * k个样本。如果size为None（默认值）并且loc和scale均为标量，那么就会返回一个值。否则会返回np.broadcast(loc, scale).size个值。
'''
mu, sigma = 0, 0.1#均值和标准差
s = np.random.normal(mu, sigma, 1000)
```

### np.random.uniform()

**np.random.uniform()：**从一个均匀分布`[low, high)`中随机采样，定义域为左闭右开。

```python
'''
numpy.random.uniform(low=0, high=1, size)
参数：
[1]low：采样下界，float类型，默认为0.
[2]high：采样上界，float类型，默认为1.
[3]size：输出样本数目，为int或元组(tuple)类型，如size=(m, n, k)则输出m * n * k个样本，缺省时输出1个值。
返回值：ndarray类型，其形状和参数size中描述一致。
'''
```



### np.array()

1、numpy array 和 python list 区别：
		python提供的**list**中，元素本质是对象，可以是任何对象，因此列表中保存的是对象的指针。如L=[1,2,3]，需要3个指针和3个整数对象，用于数值运算比较浪费内存和CPU。
		Numpy提供了**ndarray**(N-dimensional array object)对象：存储单一数据类型的多维数组。
2、np.array()

```python
np.array([1,2,3])
>>>array([1,2,3])
```

3、强制生成float类型的数组

```python
d = np.array([[1,2,3],[4,5,6]],dtype=np.float)
```

4、astype()：对array进行强制类型转换

```python
d.astype(int)
```

5、.dtype & type() 

```python
'''
type(d)：返回d的数据类型nd.array
d.dtype：返回数组中内容的数据类型
'''
```

6、array进行裁剪并按步采样

```python
'切片运算符形式：[start: end: step]，start开始索引，end结束索引，step步长，start、end、step均可省略'
arr = np.array(range(48)).reshape((2, 8, 3))
>>>arr
 [[[ 0  1  2]
  [ 3  4  5]
  [ 6  7  8]
  [ 9 10 11]
  [12 13 14]
  [15 16 17]
  [18 19 20]]

 [[21 22 23]
  [24 25 26]
  [27 28 29]
  [30 31 32]
  [33 34 35]
  [36 37 38]
  [39 40 41]]]
  
>>>arr[:, 3::2]#从索引3位置开始取值，以步长为2取到最后一个值
 [[[ 9 10 11]
  [15 16 17]]

 [[30 31 32]
  [36 37 38]]]
```

### np.mat()

1、np.mat() 与 np.array() 一样用于生成矩阵。
2、mat() 可以从字符串或列表中生成；array() 只能从列表中生成。

```python
a = np.mat(data='1,2;3,4')#mat('1,2;3,4')
b = np.array([[1,2],[3,4]])
>>
a = matrix([[1, 2],
        [3, 4]])
b = array([[1, 2],
       [3, 4]])
```

3、对生成的矩阵 matrix 和 array，矩阵乘法用 np.dot()，点乘用 np.multiply()。

### np.reshape()

1、reshape：在不改变数据内容的情况下，改变一个数组的格式。
2、numpy.reshape(a,newshape,order=‘C’)参数解释

```python
'''
a：需要处理的数组
newshape：新的格式——整数或整数数组。(1)若是整数，结果为一维数组，整数等于a中元素数量。(2)若是(-1,1)，行根据元素数量和第二个数值自动推断，任一行一列。
order：读/写顺序，{'C','F','A'},默认C
(1)'C'：横着读，横着写，优先读/写一行。
				最后一个维度变化最快，第一个维度变化最慢。类C索引顺序。
(2)'F'：竖着读，竖着写，优先读/写一列。
				最后一个维度变化最慢，第一个维度变化最快。类Fortran索引顺序。
(3)'A'：a按Fortran存储，则同'F'，否则同'C'。
				未修改a的存储方式，'A'同'C'。
				np.asfortranarray(a)：将a存储方式修改为类Fortran存储方式。
'''
```

3、reshape 两种使用方法

```python
np.reshape(a,(-1,1),order='F')
a.reshape((-1,1),order='F')
```

### np.argsort()

**np.argsort()：**返回数组值从小到大的索引值。

```python
import numpy as np
xa = np.array([3,1,2])
np.argsort(xa)
>>>array([1, 2, 0], dtype=int64)

xb = np.array([[0,3], [1,2]])
np.argsort(xb,axis=0)#按列排序
>>>array([[0, 1],
       	 [1, 0]], dtype=int64)

np.argsort(xb,axis=1)#按行排序
>>>array([[0, 1],
       	 [0, 1]], dtype=int64)
```

### np.argmax()

**np.argmax()：**用于返回一个numpy数组中最大值的索引值。当一组中同时出现几个最大值时，返回第一个最大值的索引值。
**注：**（1）在运算中，相当于剥掉一层中括号，返回一个数组，分为一维和多维。一维数组剥掉一层中括号后成了一个索引值，是一个数。而n维数组剥掉一层中括号后，返回一个 n-1 为数组，而剥掉哪层中括号，取决于axis的取值。
		（2）n维数组axis的取值从0到n-1，其对应的括号层数为从最外层向内递进。

```python
#一维数组
one_dim_array = np.array([1,4,5,3,7,2,6])
>>np.argmax(one_dim_array)
4
#多维数组
'''
二维遵循运算后降一维的原则。axis取值为0和1，对应剥掉的中括号，将里面的内容直接按逗号分隔。
0：外层
1：内层
（或“0行1列”，axis=0则将行的每同一列元素进行大小比较，axis=1则将列的每同一行元素进行大小比较）
'''
#two_dim_array是一个2x3的矩阵，对应axis为2——0,3——1，在axis=0时，剥掉2，返回1x3的数组；在axis=1时，剥掉3，返回1x2的数组。
#axis=0时，返回值为[argmax(1,0),argmax(3,4),argmax(5,3)]
#axis=1时，返回值为[argmax(1,3,5),argmax(0,4,3)]
two_dim_array = np.array([[1,3,5],[0,4,3]])
>>np.argmax(two_dim_array, axis=0)
array([0, 1, 0], dtype=int64)
>>np.argmax(two_dim_array, axis=1)
array([2, 1], dtype=int64)
'''
三维数组计算后降维，返回一个二维数组。
一个m*n*p维的矩阵，
axis为0，舍去m，返回一个 n*p 维的矩阵
axis为1，舍去n，返回一个 m*p 维的矩阵
axis为2，舍去p，返回一个 m*n 维的矩阵
'''
three_dim_array = np.array([[[1, 2, 3, 4], [-1, 0, 3, 5]],
                            [[2, 7, -1, 3], [0, 3, 12, 4]],
                            [[5, 1, 0, 19], [4, 2, -2, 13]]])
>>np.argmax(three_dim_array, axis=0)
array([[2, 1, 0, 2],
       [2, 1, 1, 2]], dtype=int64)
>>np.argmax(three_dim_array, axis=1)
array([[0, 0, 0, 1],
       [0, 0, 1, 1],
       [0, 1, 0, 0]], dtype=int64)
>>np.argmax(three_dim_array, axis=2)
array([[3, 3],
       [1, 2],
       [3, 3]], dtype=int64)
```



### np.add()

**np.add()：**两值相加。

```python
'''
np.add(x1, x2, out=None)
参数：
x1和x2对应元素相加。
out：若为None，则即时返回结果；若为一对象n，n属性要与outputs一样，则结果输出到n。
'''
n = np.ones(5)
np.add(n, 1, out=n)
```

### np.mean()

**np.mean()：**求均值。

```python
'''
np.mean(arr, axis)
假设arr为m*n矩阵：
axis不设置值：对 m*n 个数求均值，返回一个实数。
axis=0：压缩行，对各列求均值，返回 1*n 矩阵。
axis=1：压缩列，对各行求均值，返回 m*1 矩阵。
'''
```

### np.median()

**np.median()：**计算沿指定轴的中位数。

```python
'''
np.median(a,
		  axis=None,
		  out=None,
		  overwrite_input=False,
		  keepdims=False)
参数：
[1]a：输入的数组；
[2]axis：计算哪个轴上的中位数，比如输入是二维数组，那么axis=0对应行，axis=1对应列，如果对于二维数组不指定长度，将拉伸为一维计算中位数；
[3]out：用于放置求取中位数后的数组。它必须具有与预期输出相同的形状和缓冲区长度；
[4]overwrite_input：一个bool型的参数，默认为Flase。如果为True那么将直接在数组内存中计算，这意味着计算之后原数组没办法保存，但是好处在于节省内存资源，Flase则相反；
[5]keepdims：一个bool型的参数，默认为Flase。如果为True那么求取中位数的那个轴将保留在结果中；
'''

a = np.array([[10, 7, 4], [3, 2, 1]])
>>>np.median(a)
3.5
>>>np.median(a, axis=0)
array([6.5, 4.5, 2.5])
>>>np.median(a, axis=1)
array([7., 2.])

>>>m = np.zeros(2)
>>>np.median(a, axis=1, out=m)
>>>m
array([7., 2.])
```

### np.save()

**np.save()：**以`.npy`格式将数组保存到二进制文件中。

```python
'''
np.save(file, arr, allow_pickle=True, fix_imports=True)
参数：
file：要保存的文件名称，需指定文件保存路径，若未设置，保存到默认路径，文件扩展名为.npy
arr：需要保存的数组。
'''
import numpy as np
import os
os.chdir('E:\Pythonprojects\Tensorfstudy')
arr = np.random.rand(5,5)
np.save('arraytest.npy',arr)
```

### np.load()

**np.load()：**从 `.npy`、`.npz` 或 pickled文件中，加载 arrays对象或 pickled对象
注：（1）默认情况下，数组是以未压缩的原始二进制格式保存在扩展名为`.npy`的文件中，load函数加载后返回`numpy.ndarray`对象。
		（2）load函数自动识别`.npz`文件，并且返回一个类似于字典的对象，可以通过数组名作为关键字获取数组的内容。

```python
'''
np.load(file, mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
'''
```

### np.swapaxes()

**np.swapaxes()：**直接交换数组的两个轴。
**注：**理解思路不是原元组变换为新元组后，确定各元素坐标，而是直接按照各元素坐标交换后，进行新数组重组。

```python
'''
np.swapaxes(arr, axis1, axis2)
参数：
arr：需变换的数组
axis1：第一个轴的索引
axis2：第二个轴的索引
'''
a = np.ones((2,3,4,5,6))
>>np.swapaxes(a,3,1).shape
(2, 5, 4, 3, 6)

b = np.arange(12).reshape((3,2,2))
>>np.where(b==9)
(array([2], dtype=int64), array([0], dtype=int64), array([1], dtype=int64))#(2,0,1)
>>b.shape
(3, 2, 2)

c = np.swapaxes(b,2,0)
>>np.where(c==9)
(array([1], dtype=int64), array([0], dtype=int64), array([2], dtype=int64))#(1,0,2)
>>c.shape
(2, 2, 3)
```



### np.ix_()

**np.ix_()：**能把两个一维数组 转换为 一个用于选取方形区域的索引器，原理即 输入两个数组，产生笛卡尔积的映射关系。
**实际意思**是，往np.ix()里扔进两个一维数组 [1,5,7,2], [0,3,1,2]，就能先按要求选取行，再按顺序将列排序，而不用写 [ :, [0,3,1,2]]。

```python
arr = np.arange(32).reshape((8,4))
>>>arr
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11],
       [12, 13, 14, 15],
       [16, 17, 18, 19],
       [20, 21, 22, 23],
       [24, 25, 26, 27],
       [28, 29, 30, 31]])
>>>arr[[1,5,7,2]][:,[0,3,1,2]]#先选取行，再按顺序将列排列
array([[ 4,  7,  5,  6],
       [20, 23, 21, 22],
       [28, 31, 29, 30],
       [ 8, 11,  9, 10]])
>>>arr[np.ix_([1,5,7,2],[0,3,1,2])]
array([[ 4,  7,  5,  6],
       [20, 23, 21, 22],
       [28, 31, 29, 30],
       [ 8, 11,  9, 10]])
```

### np.squeeze()

np.squeeze()：删除数组形状中的单维度条目，即把shape中为 1 的维度去掉，但对非单维的维度不起作用。

```python
'''
np.squeeze(a, axis=None)
参数：
a：要处理的数组
axis：指定需要删除的维度，但指定的维度必须为单维度，否则会报错。若axis为空，则删除所有单维度的条目。
注：（1）返回值：数组。
（2）不会修改原数组。
（3）在机器学习和深度学习中，通常算法的结果是表示向量的数组（即包含两对或以上的方括号形式[[]]），如果直接利用这个数组进行画图可能显示界面为空，故利用squeeze()函数将表示向量的数组转换为秩为1的数组，之后用matplotlib函数画图时，就可以正常显示结果。
'''

a = np.arange(6).reshape(1,6)
>>>a
array([[0, 1, 2, 3, 4, 5]])
>>>a.shape
(1, 6)
b = np.squeeze(a)
>>>b
array([0, 1, 2, 3, 4, 5])
>>>b.shape
(6,)

c = np.arange(6).reshape(2,3)
>>>c
array([[0, 1, 2],
       [3, 4, 5]])
>>>np.squeeze(c)
array([[0, 1, 2],
       [3, 4, 5]])

d = np.arange(6).reshape(1,2,3)
>>>d
array([[[0, 1, 2],
        [3, 4, 5]]])
>>>np.squeeze(d).shape
(2, 3)
```

### np.expand_dims()

**np.expand_dims()：**用于扩展数组的形状。

```python
a = np.arange(12).reshape(2,2,3)
>>>a
array([[[ 0,  1,  2],
        [ 3,  4,  5]],
       [[ 6,  7,  8],
        [ 9, 10, 11]]])
>>>a.shape
(2, 2, 3)

#np.expand_dims(a, axis=0)表示在 0 位置增加维度
b = np.expand_dims(a, axis=0)
>>>b
array([[[[ 0,  1,  2],
         [ 3,  4,  5]],
        [[ 6,  7,  8],
         [ 9, 10, 11]]]])
>>>b.shape
(1, 2, 2, 3)
#np.expand_dims(a, axis=1)表示在 1 位置增加维度
b = np.expand_dims(a, axis=1)
>>>b
array([[[[ 0,  1,  2],
         [ 3,  4,  5]]],
       [[[ 6,  7,  8],
         [ 9, 10, 11]]]])
>>>b.shape
(2, 1, 2, 3)
#np.expand_dims(a, axis=2)表示在 2 位置增加维度
b = np.expand_dims(a, axis=2)
>>>b
array([[[[ 0,  1,  2]],
        [[ 3,  4,  5]]],
       [[[ 6,  7,  8]],
        [[ 9, 10, 11]]]])
>>>b.shape
(2, 2, 1, 3)
#np.expand_dims(a, axis=3)表示在 3 位置增加维度。在(2,2,3)中插入的位置共4个，再添加会报错。
b = np.expand_dims(a, axis=3)
>>>b
array([[[[ 0],
         [ 1],
         [ 2]],
        [[ 3],
         [ 4],
         [ 5]]],
       [[[ 6],
         [ 7],
         [ 8]],
        [[ 9],
         [10],
         [11]]]])
>>>b.shape
(2, 2, 3, 1)
#np.expand_dims(a, axis=-1)表示从后往前索引，在 -1 位置增加维度
b = np.expand_dims(a, axis=-1)
>>>b.shape
(2, 2, 3, 1)
```

### np.array_equal()

**np.array_equal()：**比较两个array数组，如果两个array数组有同样的`shape`和`elements`，则返回`True`，否则返回`False`。

```python
'''
np.array_equal(a1, a2, equal_nan=False)
参数：
a1，a2：输入array
equal_nan：bool，是否比较NaN是否相等
return：bool
'''
>>>np.array_equal([1,2], [1,2])
True
>>>np.array_equal(np.array([1,2]), np.array([1,2]))
True
>>>np.array_equal([1,2], [1,2,3])
False
>>>np.array_equal([1,2], [1,4])
False
>>>a = np.array([1, np.nan])
>>>np.array_equal(a, a)
False
>>>np.array_equal(a, a, equal_nan=True)
True
```

### np.sum()

**np.sum()：**求和。

```python
import numpy as np
a = np.array([[1,2],[3,4]])
>>>sum0 = np.sum(a, axis=0)

>>>sum1 = np.sum(a, axis=1)

```

### np.where()

**np.where()：**有两种用法

1. **np.where(condition, x, y)：**三个参数情况，condition表示条件，当条件成立时，where函数返回 x，当条件不成立时，where函数返回 y。
2. **np.where(condition)：**一个参数情况，当条件成立时，where函数返回每个符合condition条件的元素的坐标，以元组`tuple`的形式。

```python
'''
np.where(condition, x, y)
当where内有三个参数时，第一个参数表示条件，当条件成立时where方法返回x，当条件不成立时where返回y。
'''
>>>a = np.random.randn(2,2)
array([[-0.55674682, -0.59206032],
       [-1.40203409,  1.60513222]])

>>>np.where(a > 0, 1, -1)
array([[-1, -1],
       [-1,  1]])
```

```python
'''
np.where(condition)
当where内只有一个参数时，那个参数表示条件，当条件成立时，where返回的是每个符合condition条件的元素的坐标,返回是以元组的形式。
'''
>>>np.where(a>0)
(array([1], dtype=int64), array([1], dtype=int64))
```

### np.append()

**np.append()：**为原始array添加一些values，并返回添加后的array，非在原array上改。

```python
'''
np.append(arr, values, axis=None)
参数：
[1]arr：需要被添加values的数组
[2]values：添加到数组arr中的值（array_like，类数组）
[3]axis：可选参数，如果axis没有给出，那么arr，values都将先展平成一维数组。注：如果axis被指定了，那么arr和values需要同为一维数组或者有相同的shape，否则报错：ValueError: arrays must have same number of dimensions。
【注】对axis的理解
（1）axis的最大值为数组arr的维数-1，如arr维数等于1，axis最大值为0；arr维数等于2，axis最大值为1，以此类推。
（2）axis=0表示沿着行增长方向添加values；axis=1表示沿着列增长方向添加values。
（3）axis=0，axis=1时同上；axis=2表示沿着图像深度增长方向添加values。

返回：
添加了values的新数组
'''

'不设置axis'
#arr，values都将先展平成一维数组,然后沿着axis=0的方向在arr后添加values
a = [1, 2, 3]
b = [4, 5]
c = [[6, 7], [8, 9]]
print(np.append(a, b))
print(np.append(a, c))
>>
[1 2 3 4 5]
[1 2 3 6 7 8 9]

'设置axis'
#arr，values同为一维数组或两者shape相同
a = [1, 2, 3]
b = [4, 5]
c = [[6, 7], [8, 9]]
d = [[10, 11], [12, 13]]
print('在一维数组a后添加values,结果如下：\n{}'.format(np.append(a, b, axis=0)))
print('沿二维数组c的行增长方向添加values结果如下：\n{}'.format(np.append(c, d, axis=0)))
print('沿二维数组c的列增长方向添加values结果如下：\n{}'.format(np.append(c, d, axis=1)))
>>
在一维数组a后添加values,结果如下：
[1 2 3 4 5]
沿二维数组c的行增长方向添加values结果如下：
[[ 6  7]
 [ 8  9]
 [10 11]
 [12 13]]
沿二维数组c的列增长方向添加values结果如下：
[[ 6  7 10 11]
 [ 8  9 12 13]]

'设置axis'
#如果arr和values不同为一维数组且shape不同，则报错：
a = [1, 2, 3]
c = [[6, 7], [8, 9]]
print(np.append(a, c, axis=0))
>>
ValueError: all the input arrays must have same number of dimensions
```

### np.single()

**np.single()：**将数据转换为单精度浮点数类型，即Float类型。

```python
'''
np.single(x=0)
参数：
[1]x：需要转换数据类型的数据，默认为0
'''
>>>np.single()
0.0
>>>type(np.single())
numpy.float32
```

### np.double()

**np.double()：**将数据转换为双精度浮点数类型，即Double数据类型，与Float类型区别。此外，Double类在Python称为float64。

```python
'''
np.double(x=0)
参数：
[1]x：需要转换数据类型的数据，默认为0
'''
>>>np.double()
0.0
>>>type(np.double())
numpy.float64
>>>np.double([1, 2])
array([1., 2.])
```

### np.zeros()

**np.zeros()：**返回一个给定形状和类型用0填充的数组。

```python
'''
np.zeros(shape, dtype=float, order='C')
参数：
[1]shape：形状
[2]dtype：数据类型，可选参数，默认为numpy.float64
[3]order：可选参数，'C'代表行优先，'F'代表列优先
'''
>>>np.zeros(0)
array([], dtype=float64)

>>>np.zeros((2,2))
array([[0., 0.],
       [0., 0.]])
```

### np.empty()

**np.empty()：**返回一个给定形状的空数组。并不是真没值，而是非零的随机数。

```python
'''
np.empty(shape, dtype=float, order='C')
参数：
[1]shape：int 或 包含int的元组tuple
[2]dtype：返回的数据类型，默认为numpy.float64，此时每个值为接近0的随机数；设为object时，如list，可返回包含None的array。
[3]order：{'C', 'F'}，'C'表示存储数据以行为先，'F'表示以列为先，默认为'C'。
'''
>>>np.empty((1,2))
array([[-8.66474761e-271, -1.89718159e-137]])

>>>np.empty((1,2), dtype=list)
array([[None, None]], dtype=object)

>>>np.empty(0)
array([], dtype=float64)
```

### np.mat()

**np.mat()：**将ndarray转换为矩阵matrix。

```python
a = np.array([[1, 2, 3], [4, 5, 6]])
>>>np.mat(a)
matrix([[1, 2, 3],
        [4, 5, 6]])
```

### np.hstack()

**np.hstack()：**按水平（按列的顺序）对数据进行堆叠。传入元组、列表，或者numpy数组，返回numpy数组。
<!--h即horizontal水平方向，v即vertical垂直方向。-->

```python
'np.hstack(tup)'
>>>a = [1, 2, 3]
>>>b = [4, 5, 6]
>>>c = [[1], [2], [3]]
>>>d = [[4], [5], [6]]

>>>np.hstack((a, b))
array([1, 2, 3, 4, 5, 6])
>>>np.hstack((c, d))
array([[1, 4],
       [2, 5],
       [3, 6]])
```

### np.vstack()

**np.vstack()：**按垂直（按行的顺序）对数据进行堆叠。传入元组、列表，或者numpy数组，返回numpy数组。

```python
'np.vstack(tup)'
>>>a = [1, 2, 3]
>>>b = [4, 5, 6]
>>>c = [[1], [2], [3]]
>>>d = [[4], [5], [6]]
>>>np.vstack((a, b))
array([[1, 2, 3],
       [4, 5, 6]])
>>>np.vstack((c, d))
array([[1],
       [2],
       [3],
       [4],
       [5],
       [6]])
```

### np.poly1d()

**np.poly1d()：**给定系数构造多项式。

```python
'''
np.poly1d(c_or_r, r=False, variable=None)
参数：
[1]c_or_r：为list或array等，表示系数或根，c为coef、coefficients系数，r为roots根
[2]r：默认为False，即c_or_r表示系数，则多项式根据系数构造；若r=True，即c_or_r表示根，则多项式根据根构造
[3]variable：设置表示未知量的字母，默认为None，即x
'''
f1 = np.poly1d([2, 3, 4])
print(f1)
>>2x^2 + 3x + 4

f2 = np.poly1d([2, 3, 4], r=True, variable='z')
print(f2)
>>1z^3 - 9z^2 + 26z - 24#(z-2)(z-3)(z-4)

'常用操作'
>>>f1(2)#计算多项式的值
18

>>>f1([1, 2, 3, 4])#分别计算x=1\2\3\4的多项式值
array([ 9, 18, 31, 48])

>>>f1.c#返回生成多项式的系数
array([2, 3, 4])

>>>f2.r#返回生成多项式的根，即当多项式=0时，此等式的根
array([4., 3., 2.])

>>>f1.order#返回最高项的次方数

>>>f1[0]#返回第0项的系数，从低次向高次计数
4

>>>print(f1.deriv(1))#f1.deriv(m)表示求导，参数m表示求几次导数
4x + 3

>>>print(f1.integ(2, 2))#f1.integ(m, k)表示不定积分，参数m表示积几次不定积分，k表示积分后的常数项的值
1/6 x^4 + 1/2 x^3 + 2 x^2 + 2x + 2
```

### np.polyval()

**np.polyval()：**计算多项式在某点的值

```python
'''
np.polyval(p, x)
参数：
[1]p：array_like 或 poly1d object，array_like指1D array的多项式系数（包括常数项）。
[2]x：指定点
'''
>>>np.polyval(f1, 2)
18
>>>np.polyval([2, 3, 4], 2)
18
```

### np.polysub()

**np.polysub()：**两个多项式做差运算

```python
'''
np.polysub(a1, a2)
参数：
[1]a1&a2：array_like 或 poly1d object，array_like指1D array的多项式系数（包括常数项）。
返回：
ndarray or poly1d，为(a1 - a2)多项式差的系数
【注】polyval，polydiv，polymul，polyadd同理
'''
>>>np.polysub([2,3], [1,2])#(2x + 3) - (x + 2) = x + 1
array([1, 1])
>>>np.polysub(f2, f1)#(1z^3 - 9z^2 + 26z - 24) - (2x^2 + 3x + 4)
poly1d([  1., -11.,  23., -28.])#1x^3 -11x^2 + 23x -28
```

### np.polymul()

**np.polymul()：**两个多项式做乘运算

### np.polyadd()

**np.polyadd()：**两个多项式做和运算

### np.polyfit()

**np.polyfit()：**对一组数据进行最小二乘多项式拟合

```python
'''
np.polyfit(x, y, deg,)
参数：
[1]x：自变量x，为array_like，shape( M, )
[2]y：因变量y，为array_like，shape( M, ) or (M, K)，x和y对应关系为( x[i], y[i])
[3]deg：多项式最高项次数，为int
返回：
ndarray，shape(deg + 1,) or (deg + 1, K)，为多项式系数及最高项的次数
【注】
（1）根据给定的 点(x,y) 及 指数deg 去拟合 p(x) = p[0] * x**deg + ... + p[deg] 多项式，以['deg', 'deg-1',..., '0']的顺序返回使平方误差最小化的系数p 向量。
（2）在v1.4版本后，常用numpy.polynomial生成Polynomial类，使用numpy.polynomial.polynomial.Polynomial.fit去拟合。
'''
import matplotlib.pyplot as plt
import numpy as np

def f(x):
  return np.sin(x) + 0.5 * x
#构建噪声数据xu，yu
xu = np.random.rand(50) * 4 * np.pi - 2 * np.pi
yu = f(xu)
#用噪声数据xu、yu，得到拟合多项式系数，最高项次数为5
p = np.polyfit(xu, yu, 5)
#计算多项式的函数值，返回在x处多项式的值，p为多项式系数，元素按多项式降幂排序
py = np.polyval(p, xu)
#创建自定义画像，画像大小(8, 4)英寸
plt.figure(figsize=(8, 4))
#原先数据绘制
plt.plot(xu, yu, 'b^', label='f(x)')#b表示蓝色，^标点
#多项式拟合数据绘制
plt.plot(xu, py, 'r.', label='regression')#r表示蓝色，.标点
#标签位置
plt.legend(loc=0)
plt.show()
```

![np.polyfit()](E:\Python&Algorithm\Dat&AlgorithmNote_image\np.polyfit().png)

### np.min()

np.min()：返回ndarray中的最小值。实际调用的是amin()函数。

```python
'默认为axis=None，将array展平后计算全体最小值，而如果设置axis=0，则计算每一列的最小值；axis=1，则计算每一行的最小值。'
arr = np.array([[1, 3, 5], [7, 9, 11]])
>>>np.min(arr)
1
>>>np.min(arr, axis=0)#0行1列，0以行形式返回，1以列形式返回
array([1, 3, 5])
>>>np.min(arr, axis=1)
array([1, 7])
```

### np.max()

np.max()：返回ndarray中的最大值。实际调用的是amax()函数。

```python
arr = np.array([[1, 3, 5], [7, 9, 11]])
>>>np.max(arr)
11
>>>np.max(arr, axis=0)
array([ 7,  9, 11])
>>>np.max(arr, axis=1)
array([ 5, 11])
```

### np.median()

np.median()：返回ndarray中的中位数。

```python
arr = np.array([[1, 3, 5], [7, 9, 11]])
>>>np.median(arr)
6.0
```

### np.mean()

np.mean()：返回ndarray中的平均值。

```python
arr = np.array([[1, 3, 5], [7, 9, 11]])
>>>np.mean(arr)
6.0
```

### np.std()

np.std()：返回ndarray中的标准差。

```python
arr = np.array([[1, 3, 5], [7, 9, 11]])
>>>np.std(arr)
3.415650255319866
```

### np.var()

np.var()：返回ndarray中的方差。

```python
arr = np.array([[1, 3, 5], [7, 9, 11]])
>>>np.var(arr)
11.666666666666666
```

### np.argmin()

np.argmin()：返回ndarray中最小值的索引。

```python
arr = np.array([[1, 3, 5], [7, 9, 11]])
>>>np.argmin(arr)
0
>>>np.argmin(arr, axis=0)
array([0, 0, 0], dtype=int64)
>>>np.argmin(arr, axis=1)
array([0, 0], dtype=int64)
```

### np.argmax()

np.argmax()：返回ndarray中最大值的索引。

```python
arr = np.array([[1, 3, 5], [7, 9, 11]])
>>>np.argmax(arr)#将arr展平后的索引
5
>>>np.argmax(arr, axis=0)
array([1, 1, 1], dtype=int64)
>>>np.argmax(arr, axis=1)
array([2, 2], dtype=int64)
```



### .take()

**.take()：**沿轴返回给定位置索引中的元素，`numpy array`对象、`pandas Series`对象、`pandas DataFrame`对象、`tensorflow BatchDataset`均有该函数。注意，`tensorflow BatchDataset`是从索引'1'开始有数据，即`.take(1)`获得第一个元素。

```python
'''
a.take(indices, axis = None, convert=None, is_copy=True)
参数：
[1]a：为array、Series或DataFrame
[2]indices：一个整数数组，要取的位置
[3]axis：所选元素的轴。0表示选择行，1表示选择列。
[4]convert：是否将负指数转换为正指数。
[5]is_copy：是否返回原始对象的副本。
'''
a = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
>>>a.take(1,1)
array([2, 5，8,11])

d = pd.DataFrame(a,index=[2,3,1,0],columns=[2,0,1])
>>>d.take([0,2],axis=1)#不是根据行/列名，而是根据行/列号
    2   1
2   1   3
3   4   6
1   7   9
0  10  12
```



### .flat

**.flat：**将数组转换为一维迭代器，flat函数返回的是一个迭代器，类型为numpy.flatiter，可以用for访问迭代器每一个元素。

```python
a = np.arange(12).reshape(2,3,2)
b = a.flat
>>a
array([[[ 0,  1],
        [ 2,  3],
        [ 4,  5]],
       [[ 6,  7],
        [ 8,  9],
        [10, 11]]])
>>list(b)#用list进行输出查看
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
>>a.flat[3]
3
```



### .shape

**.shape：**ndarray的属性，获得数组的（行数，列数）。
**注：**`.shape`后面没有括号。

```python
import numpy as np
x = np.arange(4).reshape(2,2)
>>x.shape
(2,2)
```

### .size

**.size：**获得数组中元素的个数。

```python
>>x.size
4
```

### .ndim

**.ndim：**获得数组的维数。

```python
>>x.ndim
2
```

### .dtype

**.dtype：**获得数组中元素的数据类型。

```python
>>x.dtype
dtype('int32')
```

### .transpose()

**x.transpose()：**x为ndarray，将矩阵的维度进行交换。

```python
'''
(1)序列第一个"[]"为0轴，第二个为1轴，以此类推。
(2)transpose在不指定参数时，默认时矩阵转置。
'''
A.transpose((0,1))#保持A不变
A.transpose((1,0))#将0轴和1轴交换
A.transpose((0,1,2))#保持A不变
A.transpose((1,0,2))#将0轴和1轴交换
```

### .tolist()

**x.tolist()：**x为ndarray或matrix，将数组或者矩阵转换为列表。

```python
a = [[1, 2, 3], [4, 5, 6]]
b = np.array(a)
c = np.mat(a)
>>>b
array([[1, 2, 3],
       [4, 5, 6]])
>>>c
matrix([[1, 2, 3],
        [4, 5, 6]])
>>>b.tolist()
[[1, 2, 3], [4, 5, 6]]
>>>c.tolist()
[[1, 2, 3], [4, 5, 6]]
```



## sklearn

### fit() & transform() & fit_transform() & inverse_transform()

sklearn数据预处理中的**fit()、transform()、fit_transform()、inverse_transform()**

```python
#fit(trainData)：求训练集的均值、方差、最大值、最小值等训练集固有属性，可理解为一个训练拟合过程。
#transform(trainData)：在fit()的基础上，进行标准化、降维、归一化等操作（具体看使用哪个工具，如PCA、StandardScaler等）
#fit_transform(trainData)：fit()和transform()的组合，包含训练及转换。
#inverse_transform(trainData)：将transform()后的数据转换为原始数据。
"""
对trainData:
fit_transform(trainData)先对trainData进行拟合fit，获得trainData的整体指标，如均值、方差、最大值、最小值（根据具体的转换目的），然后对trainData进行transform，实现数据的标准化、归一化等。
对testData：
根据对trainData进行fit的整体指标，对testData使用同样的整体指标进行转换transform(testData)，从而保证train、test处理方式一样。
"""
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit_transform(X_train)
sc.transform(X_test)
```

### LinearRegression()

**sklearn.linear_model.LinearRegression()**

### LogisticRegression()

**sklearn.linear_model.LogisticRegression()**

```python
'''
solver：即使用的优化器，lbfgs：拟牛顿法，sag：随即梯度下降
打印报告：print(classification_report(y_test, y_pred))
'''
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver = 'lbfgs', max_iter=100)
```

### Pipeline()

1、**Pipeline：**一条龙服务、综合解决方案、流水线；对于算法或大数据分析里，可重复使用，针对新的数据，直接输入数据，可以获得结果。
2、**sklearn.pipeline.Pipeline()**

```python
"""
(1)Pipeline(Classifier)的输入对应数据挖掘的一连串步骤，其中最后一步必须是估计器Estimator，前几步为转换器Transformer。
(2)输入的数据集经过一层转换器处理后，输出结果作为下一层的输入。最后，位于Pipeline最后一层的估计器对数据进行分类。
(3)每一步用元组('名称'，步骤)来表示。
(4)Pipeline功能：
		跟踪记录各步骤的操作（以方便重现实验结果）
		对各步骤进行一个封装
		确保代码的复杂程度不至于超出掌控范围
(5)当执行pipe.fit(x_train, y_train)时，首先由StandardScaler在训练集上执行fit和transform，transformed后的数据被传递给Pipeline对象的下一步，即PCA。PCA执行fit和transform，最终将转换后的数据传递LogisticRegression，执行fit。
"""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([('sc',StandardScaler()),
                ('pca',PCA(n_components=2)),
                ('clf',LogisticRegression(random_state=1))
                ])
pipe.fit(x_train, y_train)#训练集训练模型，解释参上
pipe.predict(x_test)#验证集验证模型，返回预测值y_predict，predict()和score()可单独使用
pipe.score(x_test, y_test)#将真实值y_test与预测值y_predict进行比较，返回测试集的精度accuracy
```

### PCA()

1、PCA一般步骤：先对原始数据零均值化，然后求协方差矩阵，对协方差矩阵求特征向量和特征值，这些特征向量组成新的特征空间。
2、**sklearn.decomposition.PCA()**

```python
from sklearn.decomposition import PCA
PCA(n_components=None,copy=True,whiten=False)
"""
参数：
n_components:
意义：PCA所要保留的主成分个数n，即保留下来的特征个数n
类型：int或string，缺省时默认None，所有成分被保留
		 n_components = 1，将原始数据降到一个维度。
		 n_components = 'mle'，将自动选取特征个数n，使满足所要求的方差百分比。

copy：
意义：是否在原始数据的副本上运算，True则原始数据的不做任何改变，在副本上运算；False则在原始数据上降维计算。
类型：bool，缺省时默认True

whiten：
意义：白化，使得每个特征具有相同的方差。
类型：bool，缺省时默认False
PCA属性：
components_：返回具有最大方差的成分
explained_variance_ratio_：返回所保留的n个成分各自的方差百分比
n_components_：返回所保留的成分个数n
"""
```

### LabelEncoder()

**sklearn.preprocessing.LabelEncoder()：**将离散型数据转换成0到n-1间的数。n是一个列表不同取值的个数，或某个特征所有不同取值的个数。

```python
from sklearn.preprocessing import LabelEncoder
y = ['A','B','C','A']
encoder = LabelEncoder()
y = encoder.fit_transform(y)
#y为array([0,1,2,0],dtype=int64)
```

### StandardScaler()

**sklearn.preprocessing.StandardScaler**
1、sklearn之数据预处理——StandardScaler归一化
2、**归一化原因：**
		a、归一化可提高梯度下降求最优解的速度
		b、归一化可提高精度
3、不需要与需要做归一化的机器学习算法：
		**不需要：**概率模型（树形模型）【因为它们不关心变量的值，只关心变量的分布和变量之间的条件概率】，如决策树、RF。
		**需要：**Adaboost、SVM、LR、KNN、KMeans等最优化问题。
4、StandardScaler作用：去均值和方差归一化。且是针对每一个特征维度来做，而非针对样本。
5、StandardScaler做法：标准差标准化，使得经过处理的数据符合标准正态分布，即均值为0，标准差为1。
6、使用方法

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
np.random.seed(0)
a = np.random.randint(2,5,size=(3,3))
scaler = StandardScaler()
newa = scaler.fit_transform(a)
scaler.mean_#返回均值
scaler.var_#返回方差
np.sqrt(scaler.var_)#返回标准差
```

### train_test_split()

**sklearn.model_selection.train_test_split()**

```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size,random_state)
'''
参数：
test_size：若在0~1间，为测试集样本数与原始样本数之比；若为整数，为测试集样本的数目。
random_state：随机数的种子。可以为整数、RandomState实例或None，默认为None。
			(1)若为None，每次生成的数据都是随机的，可能不一样。
			(2)若为整数，每次生成的数据都相同。
注：
随机数种子：是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。比如random_state设为1，其他参数一样的情况下，划分得到的随机数组是一样的，而不填或填0，每次划分都会不一样。
'''
```

### make_classification()

**sklearn.datasets.make_classification()**：生成一个n类随机分类问题。

```python
'''
make_classification(n_features, n_classes)
参数：
n_features特征数：用来区分每一类，每类有独属的特征集。
n_classes类数：用来区分每个样本，每个样本都是n类中一类。
'''
```

### PolynomialFeatures()

**sklearn.preprocessing.PolynomialFeatures()：**使用多项式进行特征的构造。
**Tips：**可与线性回归LinearRegression结合进行多项式回归。

```python
'''
PolynomialFeatures(degree, interaction_only, include_bias)
参数：
degree：控制多项式的次数，默认2次多项式。
interaction_only：默认为False；若为True，则无特征自己与自己结合的项，如a^2。
include_bias：默认为True，表示包含项'1'。
'''
from sklearn.preprocessing import PolynomialFeatures
ply = PolynomialFeatures()
fea = [[1,2,3]]#一维对象看为一组特征
fea1 = np.array([1,2,3])
fea_a = fea1[:,np.newaxis]
fea_b = fea1[np.newaxis,:]
res = ply.fit_transform(fea)
a = ply.fit_transform(fea_a)
b = ply.fit_transform(fea_b)
>>
res= array([[1., 1., 2., 3., 1., 2., 3., 4., 6., 9.]])
fea1= array([1,2,3])
fea_a= array([[1],
       			  [2],
       			  [3]])
fea_b= array([[1,2,3]])
a= array([[1., 1., 1.],
       		[1., 2., 4.],
       		[1., 3., 9.]])
b= array([[1., 1., 2., 3., 1., 2., 3., 4., 6., 9.]]) 
```

### cross_val_score()

**sklearn.model_selection.cross_val_score()：**(1)分别在K-1折上训练模型，在余下的1折上验证模型，并保存余下1折中的预测得分。(2)该函数得到K折验证中每一折的得分，K个得分取平均值即模型的平均性能，该平均性能可作为模型的泛化性能参考。
**Tips：**交叉验证法的作用是尝试利用不同的训练集/测试集划分来对模型做多组不同的训练/测试，来应对测试结果过于片面，以及训练数据不足的问题。

```python
'''
cross_val_score(estimator, X, y=None, scoring=None, cn=None, n_jobs=1)
参数：
estimator：估计方法对象(分类器)。
X：数据特征(Features)。
y：数据标签(Labels)。
scoring：调用方法(包括 accuracy 、 mean_squared_error 和 neg_mean_squared_error 等等)。
cv：几折交叉验证。
n_jobs：同时工作的cpu个数(-1代表全部)。
'''
```

### RandomizedSearchCV()

**sklearn.model_selection.RandomizedSearchCV()：**超参数调优方法之随机搜索。

```python
'''
sklearn.model_selection.RandomizedSearchCV(estimator, 
																					param_distributions, 
																					n_iter, 
																					scoring,
																					cv,)
参数：
[1]estimator：进行超参数寻优的模型。
[2]param_distributions：寻优的超参数及可取的值，为字典或字典列表。
	[{需要调整的超参数1:超参数可能的值}, {需要调整的超参数2:超参数可能的值}, ]
[3]n_iter：采样的参数设置数，n_iter用于在运行时间与结果质量间权衡，为int，默认为10.
[4]scoring：误差函数，默认为None。如果为None，则使用estimator本身的误差函数。
[5]cv：交叉验证参数，决定交叉验证拆分策略，默认为None。当为None，则默认使用 5折交叉验证（在0.22版本将3-fold改为5-fold）；当为整数K，则使用K折交叉验证。
返回值属性：
[1]best_estimator_：返回寻优到的最优estimator
[2]best_score_：返回寻优到的最佳分数
[3]best_params_：返回寻优到的最佳参数
'''
import keras
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV

def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[8]):
  model = keras.models.Sequential()
  model.add(keras.layers.InputLayer(input_shape=input_shape))
	for layer in range(n_hidden):
    model.add(keras.layers.Dense(n_neurons, activation="relu"))
  model.add(keras.layers.Dense(1))
  optimizer = keras.optimizers.SGD(lr=learning_rate)
  model.compile(loss="mse", optimizer=optimizer)
  return model

param_distribs = {
  "n_hidden": [0, 1, 2, 3],
  "n_neurons": np.arange(1, 100),
  "learning_rate": reciprocal(3e-4, 3e-2),
}

rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3)
rnd_search_cv.fit(X_train, y_train, epochs=100, 
                  validation_data=(X_valid, y_valid),
                  callbacks=[keras.callbacks.EarlyStopping(patience=10)]
                 )
>>>rnd_search_cv.best_params_
>>>rnd_search_cv.best_score_
>>>model = rnd_search_cv.best_estimator_.model
```



### datasets

**sklearn.datasets**

#### load_iris



## matplotlib

### plt.setp()

**matplotlib.pyplot.setp()：**设置对象属性或属性的取值要求。

```python
'''
matplotlib.pyplot.setp(obj,*args,**kwargs)
参数：
obj：需要设置属性的对象或对象列表；类型为Artist对象或Artist对象列表，即matplotlib所有可见对象；必备参数。
file：当查询属性取值要求时输出文件的位置；类文件对象；默认值为sys.stdout。
*args、**kwargs：需要设置的属性值。
'''
#设置多个对象的一个属性
import matplotlib.pyplot as plt
#lines包含两条线段x1=[1,2,3],y1=[1,2,3];x2=[4,5,6],y2=[4,5,6]，两条线段在同一张图上。
lines = plt.plot([1,2,3],[1,2,3],[4,5,6],[4,5,6])
plt.setp(lines,linestyle='--')
```

### plt.plot()&ax.plot()

**matplotlib.pyplot.plot()**
【Tips】将`Matplotlib`绘图和平常画画类比，可以把`Figure`想象成一张纸（一般称之为画布），`Axes`代表的则是纸中的一片区域（可以有多个区域，即`subplots`）。

```python
'''
matplotlib.pyplot.plot(x,y,format_string,**kwargs)
参数：
[1]x：x轴数据，列表或数组，可选。
[2]y：y轴数据，列表或数组，必备参数。
[3]format_string：控制曲线的格式字符串，由颜色字符、风格字符和标记字符组成，可选，参见下图。
[4]rot：横坐标旋转角度。
[5]marker：图上画点的地方标上符号，'.'、'*'、'o'、'^'等
[6]markersize：marker符号的大小，浮点数
【注】颜色和线性设置格式：plot(x, y, "r.")
'''
'第一种方式'
#该方式先生成一个Figure画布，然后在这个画布上隐式生成一个画图区域进行画图
plt.figure()
plt.plot([1,2,3], [4,5,6])

'第二种方式'
#该方式同时生成了Figure和axes两个对象，然后用ax对象在其区域内进行绘图。
#生成的 fig 和 ax 分别对画布 Figure 和绘图区域 Axes 进行控制
fig, ax = plt.subplots()
ax.plot([1,2,3],[4,5,6])

'绘制多项式曲线图'
p = np.poly1d([2,3,4])
x = np.linspace(1, 5, 200)
y = np.polyval(p, x)
plt.figure(figsize=(8, 6))
plt.plot(x, y, "r-", label= "2x^2 + 3x + 4")
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc=0)
plt.show()
```

![颜色字符](E:\Python&Algorithm\Dat&AlgorithmNote_image\颜色字符.png)
![标记字符](E:\Python&Algorithm\Dat&AlgorithmNote_image\标记字符.png)

<img src="E:\Python&Algorithm\Dat&AlgorithmNote_image\y=2x^2+3x+4.png" alt="y=2x^2+3x+4" style="zoom:80%;" />
### plt.subplot()

**matplotlib.pyplot.subplot()：**划分画布，并创建单个子图对象。

```python
"""
matplotlib.pyplot.subplot(nrows, ncols, index)
[1]子图位置由1开始。
[2]也可写成plt.subplot(2,2,1)
[3]第一个参数代表行数，第二个参数代表列数，第三个参数代表第几个子图位置
"""
import matplotlib.pyplot as plt
plt.subplot(221)#将整个画布分割成2*2个子图像，当前子图位置为1。
plt.subplot(222)#表示将画布Figure分为2行2列，当前位置为2
plt.subplot(223)
plt.subplot(224)#表示将画布Figure分为2行2列，当前位置为4
```

### plt.subplots()

**matplotlib.pyplot.subplots()：**划分画布，并一次性创建多个子图对象。
【Tips】`fig, axes = plt.subplots(23)`：表示一次性在figure上创建2*3的网格，而使用`plt.subplot()`只能一个一个的添加，如下：

```python
fig = plt.figure()
ax = plt.subplot(231)
ax = plt.subplot(232)
ax = plt.subplot(233)
```

```python
fig, ax = plt.subplots()
'等价于'
fig, ax = plt.subplots(11)
'等价于'
fig = plt.figure()
ax = plt.subplot(111)
```

```python
fig, ax = plt.subplots(nrows=1, ncols=2)#ax对应的是一个列表，存储了两个Axes对象
ax[0].plot([1, 2, 3], [4, 5, 6])
ax[1].scatter([1,2,3], [4, 5, 6])#散点图
```

```python
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot([1, 2, 3], [4, 5, 6])
plt.show()
```



### plt.figure()

**matplotlib.pyplot.figure()：**创建自定义图像。

```python
'''
figure(num=None, figsize=None, dpi=None, facecolor=None, edgecolor=None, frameon=True)
参数：
[1]num：图像编号或名称，数字为编号，字符串为名称。
[2]figsize：指定figure的宽和高，单位为英寸。
[3]dpi：指定绘图对象的分辨率，即每英寸多少个像素，缺省值为80。
[4]facecolor：背景颜色。
[5]edgecolor：边框颜色。
[6]frameon：是否显示边框。
'''
import matplotlib.pyplot as plt
fig = plt.figure(figzie=(4,3), facecolor='blue')
#figsize=(4,3)表示宽4英寸，高3英寸，结合dpi，得大小为320*240像素。
```

### plt.savefig()

**matplotlib.pyplot.savefig()：**让程序自动将图表保存到文件。

```python
plt.savefig(fname, dpi=300)#fname为输出文件名
```

### plt.imshow()

**matplotlib.pyplot.imshow()：**实现热图绘制。

```python
'''
plt.imshow(X, cmap=None,)
参数：
X：要绘制的图像或数组。
cmap：颜色图谱，默认绘制RGB(A)颜色空间。
'''
```

### plt.tight_layout()

**matplotlib.pyplot.tight_layout()：**tight_layout()会自动调整子图参数，使之填充整个图像区域，它仅仅检查坐标轴标签、刻度标签以及标题的部分，tight_layout()也会调整子图之间的间隔来减少堆叠。

```python
...#绘图代码
plt.tight_layout()
```

### plt.legend()

**matplotlib.pyplot.legend()：**`legend()`函数会给图加上图例。

```python
'''
plt.legend(
			handles, 
			labels,
			loc='best',
			fontsize,
			title=None,
			)
参数：
[1]handles：画图例的对象
[2]labels：各图例的名字，handles和labels是相对应的
[3]loc：图例的位置，默认"best"
[4]fontsize：图例字体大小
[5]title：图例标题，默认None
'''
ax.legend([line1, line2, line3], ['label1', 'label2', 'label3'])

line1, = ax.plot([1, 2, 3], label='label1')
line2, = ax.plot([1, 2, 3], label='label2')
ax.legend(handles=[line1, line2])
```

### plt.xlim()&plt.ylim()

**plt.xlim()&plt.ylim()：**该函数用于设置 x 或 y 轴坐标范围。

```python
plt.xlim(1, 10)
plt.ylim(0, 1)
```

### plt.text()

**plt.text()：**画图时，给图中的点加标签。

```python
'''
plt.text(x, y, s, fontsize, verticalalignment, horizontalalignment, rotation, kwargs)
参数：
[1]x&y：标签添加的位置。注释文本内容所在位置的横/纵坐标，默认是根据坐标轴的数据来度量，为绝对值，即图中点所在位置的对应的值。特别的，如果要变换坐标系，要用到transform=ax.transAxes参数。
[2]s：标签的符号，字符串格式。
[3]fontsize：标签的字体大小，int
[4]verticalalignment：垂直对齐方式，可选'center'，'top'，'bottom'，'baseline'等
[5]horizontalalignment：水平对齐方式，可选'center'，'right'，'left'等
[6]rotation：标签的旋转角度，以逆时针计算，int
[7]family：设置字体
[8]style：设置字体的风格
[9]weight：设置字体的粗细
[10]bbox：给字体添加框，如bbox=dict(facecolor='red', alpha=0.5)
[11]string：注释文本内容
[12]color：注释文本内容的字体颜色
'''
'显示每个点的坐标，添加以下语句：'
for a, b in zip(x, y):#添加该循环显示坐标
  plt.text(a, b, (a, b), ha='center', va='bottom', fontsize=10)
```



## pandas

### pd.DataFrame()

**pandas.DataFrame()：**创建DataFrame，它是一种数据结构，一种二维表。

```python
'''使用numpy函数创建'''
#index行名，columns列名
#后面两个参数可以使用list输入，list的长度要和DataFrame的大小匹配。
#index和columns这两个参数是可选的，可以选择不设置，也可以设置为一样。
>>>pd.DataFrame(np.random.randn(3,3), index=list('abc'), columns=list('ABC'))
          A         B         C
a -0.433073 -1.471495  0.097367
b -0.225139 -0.001251 -1.242341
c  0.490311 -0.347914  2.094115

'''直接创建'''
#第一个参数必须是二维，如[[1,2,3]]维度是(1,3)，而[1,2,3]维度是(3,)，所以使用[1,2,3]会报错。
#list维度查看：np.array([[1,2,3]]).shape，list不能使用.shape查看维度。
>>>pd.DataFrame([[1,2,3]], columns=list('ABC'))
   A  B  C
0  1  2  3

'''使用字典创建'''
'两个Series或list可以用字典合并在一起并赋列名'
dic = {
    'a':[1,2,3],
    'b':[4,5,6],
    'c':[7,8,9]
}
df = pd.DataFrame(dic)
>>
   a  b  c
0  1  4  7
1  2  5  8
2  3  6  9

'在DataFrame中新添加一列，直接指明列名，然后赋值'

data = pd.DataFrame(columns=['a','b'], data=[[1,2],[3,4]])
>>> data
   a  b
0  1  2
1  3  4

data['c'] = ''
>>> data
   a  b c
0  1  2
1  3  4

data['d'] = [5,6]
>>> data
   a  b c  d
0  1  2    5
1  3  4    6

#以下将报错
data['e'] = []
```

### pd.read_excel()

**pd.read_excel()：**通过io和sheet_name读取Excel表，返回DataFrame。

```python
'''
pd.read_excel(io,
			  sheet_name=0,
			  header=0
			  engine=None,
			  na_values=None,
			  keep_default_na=True
			  )
参数：
[1]io：文件类对象，地址
[2]sheet_name：指定加载的表，默认为0
[3]header：指定表头行号
[4]engine：指定Excel处理引擎，如果io不是buffer或path，该参数必须指定，可选的值有：None，"xlrd"，"openpyxl"，"odf"或"gbk"
[5]na_values：指定NA数据
[5]keep_default_na：处理数据是否包含默认的NaN值。
   若keep_default_na为True，且na_value被指定，na_values被添加到默认NaN用于数据处理；
   若keep_default_na为False，且na_value未被指定，则只使用默认NaN来处理数据；
   若keep_default_na为False，且na_value被指定，则只使用默认NaN来处理数据；
   若keep_default_na为False，且na_value未被指定，则NaN会为空字符串；
'''
```

### pd.read_csv()

**pd.read_csv()：**通过文件路径读取Excel表。

```python
'''
pd.read_csv(filepath_or_buffer,
				sep,
				header,
				names,
				index_col,
				parse_dates,
				date_parser,
				)
参数：
[1]filepath_or_buffer：访问文件的有效路径，可为URL。
[2]sep：指定分隔符。如果不指定参数，则会尝试使用','分隔。
[3]header：表头，设置DataFrame的列名称。若不设置header，则将读取的第一行作为列索引；若header=None，即指明原始文件数据没有列索引，则会自动加上列索引；若header=0，表示将读取的第一行作为列索引，这时设置names会替换掉该索引；若header=1，则选取第二行作为列索引，第二行下面的为数据。
[4]names：当names未赋值时，header默认为0，即选取数据文件的第一行作为列名；当names被赋值时，header未被赋值，header默认为None；如果两者都赋值，则会实现组合功能。
[5]index_col：指定读取数据的某一列作为DataFrame的行索引。默认为None，即不指定行索引，这会自动添加行索引（0 1 2 3...）。可将index_col指定为列名或列对应的索引，则会将该列作为行索引。
[6]parse_dates：指定某些列为时间类型，这个参数一般搭配date_parser使用。
[7]date_parser：用来配合parse_dates参数，因为有的列虽然是日期，但没办法直接转化，需要指定一个解析格式。
注：
（1）一般来说，读取文件会有一个表头的，一般是第一行，但是有的文件只是数据而没有表头，那么就可以通过names手动指定、或者生成表头，而文件里面的数据则全部是内容。而names适用于这种没有表头的情况。
'''
#names被赋值，header未被赋值
pd.read_csv('o3o.csv', names=["编号", "姓名", "地址", "日期"])
#names和header均被赋值
'header=0表示第一行当作表头，下面当成数据。之后，names=[...]表示用[...]替换掉表头'
pd.read_csv('o3o.csv', header=0, names=["编号", "姓名", "地址", "日期"])
#index_col被赋值
pd.read_csv('o3o.csv', index_col="name")
pd.read_csv('o3o.csv', index_col=[0])

#使用parse_dates与date_parser
from datetime import datetime
pd.read.csv('o3o.csv', sep="\t", parse_dates=["date"], date_parser=lambda x:datetime.strptime(x, "%Y年%m月%d日"))
```

### pd.cut()

**pd.cut()：**将数据按宽度进行切分

```python
'''
pd.cut( x,
		bins,
		right=True,
		labels=None,
		retbins=False,
		precision=3,
		include_lowest=False)
参数：
[1]x：待切割的数据，为类array对象，且必须是一维。
[2]bins：为整数、序列尺度或间隔索引。若bins是一个整数，它定义了x宽度范围内的等宽面元数量，但是在这种情况下，x的范围在每个边上被延长1%，以保证包括x的最小值或最大值。如果bin是序列，它定义了允许非均匀bin宽度的bin边缘。在这种情况下没有x的范围的扩展。
[3]right：为布尔值。是否是左开右闭区间，right=True，左开右闭，right=False,左闭右开
[4]labels：用作结果箱的标签。必须与结果箱相同长度。如果FALSE，只返回整数指标面元。
[5]retbins：为布尔值。是否返回面元
[6]precision：为整数。返回面元的小数点几位
[7]include_lowest：为布尔值。第一个区间的左端点是否包含
'''
a = np.arange(20)
b = [2, 5, 9, 15, 17, 20]
df = pd.DataFrame(list(a), columns=list('a'))
df['b'] = pd.cut(df['a'], bins=b, labels=['低', '中低', '中', '中高', '高'], right=False)
>>
     a    b
0    0  NaN
1    1  NaN
2    2    低
3    3    低
4    4    低
5    5   中低
6    6   中低
7    7   中低
8    8   中低
9    9    中
10  10    中
11  11    中
12  12    中
13  13    中
14  14    中
15  15   中高
16  16   中高
17  17    高
18  18    高
19  19    高
#labels=None
df['b'] = pd.cut(df['a'], bins=b, labels=None, right=False)
>>
     a             b
0    0           NaN
1    1           NaN
2    2    [2.0, 5.0)
3    3    [2.0, 5.0)
4    4    [2.0, 5.0)
5    5    [5.0, 9.0)
6    6    [5.0, 9.0)
7    7    [5.0, 9.0)
8    8    [5.0, 9.0)
9    9   [9.0, 15.0)
10  10   [9.0, 15.0)
11  11   [9.0, 15.0)
12  12   [9.0, 15.0)
13  13   [9.0, 15.0)
14  14   [9.0, 15.0)
15  15  [15.0, 17.0)
16  16  [15.0, 17.0)
17  17  [17.0, 20.0)
18  18  [17.0, 20.0)
19  19  [17.0, 20.0)
```

### pd.qcut()

**pd.qcut()**：将数据按分位数进行切分

```python
'''
pd.qcut( x,
		 q,
		 labels=None,
		 retbins=False,
		 precision=3,
		 duplicates='raise')
参数：
[1]q：为整数或分位数数组。整数如4，表示按照四分位数进行切分。
'''
```

### pd.concat()

pd.concat()：沿着一条轴，将多个对象堆叠到一起。不仅可以指定连接的方式（outer join或inner join），还可以指定按照某个轴进行连接。
【注】（1）pd.concat()只是单纯的把两个表拼接在一起，不去重，可使用drop_duplicates方法达到去重效果。
				（2）当axis=0时，`pd.concat([obj1, obj2])`与`obj1.append(obj2)`的效果相同
				当axis=1时，`pd.concat([obj1, obj2], axis=1)`与`pd.concat(obj1, obj2, left_index=True, right_index=True, how='outer')`的效果相同。

```python
'''
pd.concat(objs,
					axis=0,
					join='outer',
					join_axes=None,
					ignore_index=False,
					keys=None,
					ignore_index=False,
					)
参数：
[1]objs：需要连接的对象集合，一般是列表或字典
[2]axis：连接轴向，默认为0
[3]join：参数为'outer'或'inner'
[4]join_axes=[]：指定自定义的索引
[5]keys=[]：创建层次化索引
[6]ignore_index：默认为False，为True时重建索引
'''
df1 = pd.DataFrame(np.random.randn(3,4), columns=['a', 'b', 'c','d'])
df2 = pd.DataFrame(np.random.randn(2,3), columns=['b','d', 'a'])
>>>pd.concat([df1, df2])
          a         b         c         d
0 -0.201097  0.074312 -0.819731 -1.561176
1 -2.834164  0.182110 -0.370617  0.207156
2 -0.006367  0.064933 -0.919151  0.937951
0 -0.557437  0.576415       NaN  0.723708
1  0.117623 -1.813843       NaN  0.747428

>>>pd.concat([df1, df2], ignore_index=True)
          a         b         c         d
0 -0.201097  0.074312 -0.819731 -1.561176
1 -2.834164  0.182110 -0.370617  0.207156
2 -0.006367  0.064933 -0.919151  0.937951
3 -0.557437  0.576415       NaN  0.723708
4  0.117623 -1.813843       NaN  0.747428
```

### pd.merge()

pd.merge()：通过键拼接列

### pd.to_datetime()

pd.to_datetime()：该函数将字符型的时间数据转换为时间型数据

```python
'''
pd.to_datetime(arg, format=None)
参数：
[1]arg：要处理的数据，对于读入的csv或excel文件而言，指字符型时间的格式列。可以转换为datetime的object。
[2]format：转换后的时间格式，例如"%d/%m/%Y"
'''
data["new_date"]=pd.to_datetime(data['date'])
data.head()
data.dtypes
```

### pd.date_range()

pd.date_range()：用于生成指定长度的日期索引，默认产生按天计算的时间点（即日期范围）。

```python
'''
pd.date_range(start=None, end=None, periods=None, freq=None, tz=None, normalize=False, name=None, closed=None, **kwargs)
参数：
[1]start：开始时间
[2]end：结束时间
[3]periods：生成的时间序列数据的长度，整数；结合start或end使用。
[4]normalize：若参数为True，表示将start、end参数值正则化到午夜时间戳
[5]name：生成时间索引对象名称
[6]freq：生成时间序列数据的频率，默认为D，即时间频率为天；另外，H为小时，T为每分钟；该频率由一个基础频率和乘数组成。基础频率为D、H、T等。
'''
'两种生成方式：（1）起始与结束时间；（2）仅有一个起始或结束时间，加上一个序列长度'
pd.date_range('20200801', '20200810')
pd.date_range(start='20200801', periods=10)
pd.date_range(end='20200810', periods=10)

'使用pd.date_range生成固定频率的日期和时间跨度序列，然后使用pandas.Series.dt提取特征'
#生成DataFrame存储器及日期
df = pd.DataFrame()
df['time'] = pd.date_range('2/5/2019', periods = 6, freq='2H')
print(df['time'])
>>
0   2019-02-05 00:00:00
1   2019-02-05 02:00:00
2   2019-02-05 04:00:00
3   2019-02-05 06:00:00
4   2019-02-05 08:00:00
5   2019-02-05 10:00:00
Name: time, dtype: datetime64[ns]
#提取特征
df['year'] = df['time'].dt.year
df['month'] = df['time'].dt.month
df['day']  = df['time'].dt.day
df['hour'] = df['time'].dt.hour
df['minute'] = df['time'].dt.minute
print(df.head(6))
>>
                 time  year  month  day  hour  minute
0 2019-02-05 00:00:00  2019      2    5     0       0
1 2019-02-05 02:00:00  2019      2    5     2       0
2 2019-02-05 04:00:00  2019      2    5     4       0
3 2019-02-05 06:00:00  2019      2    5     6       0
4 2019-02-05 08:00:00  2019      2    5     8       0
5 2019-02-05 10:00:00  2019      2    5    10       0
```



### .append()

.append()：向DataFrame对象中添加新的行，如果添加的列名不在DataFrame对象中，将会被当作新的列进行添加

```python
'''
obj1.append(obj2,
            ignore_index=False,
           	verify_integrity=False,
            sort=None)
参数：
[1]obj1&obj2：为DataFrame、series、dict、list等对象
[2]ignore_index：默认为False，为True时重建索引
[3]verify_integrity：默认为False，若为True时，当创建相同的index时会抛出ValueError的异常
[4]sort：默认为None
'''
df1 = pd.DataFrame(np.random.randn(3,4), columns=['a', 'b', 'c','d'])
df2 = pd.DataFrame(np.random.randn(2,3), columns=['b','d', 'a'])
>>>df1.append(df2, ignore_index=True)
          a         b         c         d
0 -0.201097  0.074312 -0.819731 -1.561176
1 -2.834164  0.182110 -0.370617  0.207156
2 -0.006367  0.064933 -0.919151  0.937951
3 -0.557437  0.576415       NaN  0.723708
4  0.117623 -1.813843       NaN  0.747428

>>>df1.append(df2)
          a         b         c         d
0 -0.201097  0.074312 -0.819731 -1.561176
1 -2.834164  0.182110 -0.370617  0.207156
2 -0.006367  0.064933 -0.919151  0.937951
0 -0.557437  0.576415       NaN  0.723708
1  0.117623 -1.813843       NaN  0.747428
```



### .dtypes

.dtypes：查看列的数据类型。

```python
>>>df.dtypes
a    int64
b    int64
c    int64
dtype: object
```

### .describe()

**.describe()：**描述

### .head()

**.head()：**Pandas读取数据之后使用Pandas的head()函数来观察一下读取的数据，head()函数可以查看前几行数据，默认读取前五行数据，可以自己设置。

```python
import pandas as pd
csv_path = "xxx.csv"#xxx为具体文件名
df = pd.read_csv(csv_path)
df.head()
df.head(2)#看前2行
```

### .tail()

**.tail()：**查看后几行的数据，默认五行。

```python
df.tail()
df.tail(2)
```

### .index

**.index：**查看行名。

```python
>>>df.index
RangeIndex(start=0, stop=3, step=1)
```

### .columns

**.columns：**查看列名。

```python
>>>df.columns
Index(['a', 'b', 'c'], dtype='object')
```

### .values

**.values：**可以查看DataFrame里的数据值，返回的是一个数组array。

```python
>>>df.values
array([[1, 4, 7],
       [2, 5, 8],
       [3, 6, 9]], dtype=int64)
#查看某一行所有的数据值，查看列得使用.loc或.iloc
>>>df['a'].values
array([1, 2, 3], dtype=int64)
```

### .loc[]

**.loc[]：**通过行（或列）名（Index或columns）来取行（或列）数据。
注：`.loc[]`用`[:]`切片时，为左闭右闭；`.iloc[]`用`[:]`切片时，为左闭右开。

```python
import numpy as np
import pandas as pd
data = pd.DataFrame(np.arange(16).reshape(4,4), index=list('abcd'), columns=list('ABCD'))
>>>data
    A   B   C   D
a   0   1   2   3
b   4   5   6   7
c   8   9  10  11
d  12  13  14  15
#取名为'a'的行
>>>data.loc['a']#返回值类型为Series(4,)
A    0
B    1
C    2
D    3
Name: a, dtype: int32
>>>data.loc[['a']]#返回值类型为DataFrame(1,4)
   A  B  C  D
a  0  1  2  3
#取名为'A'的列
>>>data.loc[:,'A']#返回值类型为Series(4,)
a     0
b     4
c     8
d    12
Name: A, dtype: int32
>>>data.loc[:, ['A']]#返回值类型为DataFrame(4,1)
    A
a   0
b   4
c   8
d  12

#loc切片索引为左闭右闭
>>>data.loc['a':'d']
    A   B   C   D
a   0   1   2   3
b   4   5   6   7
c   8   9  10  11
d  12  13  14  15
data1 = pd.DataFrame(np.arange(16).reshape(4,4), columns=list('ABCD'))
>>>data1
    A   B   C   D
0   0   1   2   3
1   4   5   6   7
2   8   9  10  11
3  12  13  14  15
>>>data1.loc[0:2]
   A  B   C   D
0  0  1   2   3
1  4  5   6   7
2  8  9  10  11
```

### .iloc[]

**.iloc[]：**通过行（或列）号来取行（或列）数据。
注：`.loc[]`用`[:]`切片时，为左闭右闭；`.iloc[]`用`[:]`切片时，为左闭右开。

```python
#与.loc[]类似
>>>data.iloc[0]
A    0
B    1
C    2
D    3
Name: a, dtype: int32
>>>data.iloc[:, 0]
a     0
b     4
c     8
d    12
Name: A, dtype: int32
#iloc切片索引为左闭右开
>>>data.iloc[0:3]
   A  B   C   D
0  0  1   2   3
1  4  5   6   7
2  8  9  10  11
>>>data1.iloc[0:2]
   A  B  C  D
0  0  1  2  3
1  4  5  6  7
```

### .shape

**.shape：**查看维度。

```python
>>>df.shape
(3, 3)
>>>df.shape[0]#查看行数
3
>>>df.shape[1]#查看列数
3
```

### .corr()

.corr()：返回该数据类型皮尔逊相关系数矩阵，即每两个类型直接的相关性。

### .apply()

**.apply()：**当要对数据框（DataFrame）的数据进行按行或按列操作时用`apply()`。
注：想要对数据使用函数，可借助`apply()`、`applymap()`、`map()`来应用函数，括号里面可以是直接函数式，或自定义函数（def）或者匿名函数（lambda）。

```python
df = DataFrame({
    			"sales1":[-1,2,3],
    			"sales2":[3,-5,7],
				})
>>
   sales1  sales2
0      -1       3
1       2      -5
2       3       7
>>>df.apply(lambda x : x.max() - x.min(), axis=1)#axis=1表示处理结果为一列
0    4
1    7
2    4
dtype: int64
```

### .applymap()

**.applymap()：**当要对数据框（DataFrame）的每一个数据进行操作时用`applymap()`，返回DataFrame。

```python
>>>df.applymap(lambda x : 1 if x>0 else 0)
   sales1  sales2
0       0       1
1       1       0
2       1       1
```

### .map()

**.map()：**当要对Series的每一个数据进行操作时用`map()`。

```python
>>>df.sales1.map(lambda x : i if x>0 else 0)
0    0
1    1
2    1
Name: sales1, dtype: int64
```

注：当要对数据进行应用函数时，先看数据结构是DataFrame还是Series，Series使用`map()`，DataFrame每个元素使用`applymap()`，DataFrame按行或列使用`apply()`。

### .dropna()

**.dropna()：**删除DataFrame中包含缺失值的行或列。

```python
'''
DataFrame.dropna( axis=0,
				  how='any',
				  inplace=False)
参数：
[1]axis：axis=0或axis='index'删除含有缺失值的行；axis=1或axis='columns'删除含有缺失值的列。默认为0。
[2]how：how='all'表示删除全是缺失值的行（列）；how='any'表示删除只要含有缺失值的行（列）。默认为'any‘。
[3]thresh：thresh=n表示保留至少含有n个非na数值的行。
[4]subset：表示要在哪些列中查找缺失值。
[5]inplace：默认为False，False表示创建一个副本，修改副本，原对象不变；True表示直接修改原对象。
'''
df = pd.DataFrame({"name": ['A', 'B', 'C'],
                   "toy": [np.nan, 'D', 'E'],
                   "born": [pd.NaT, pd.Timestamp("1940-04-25"), pd.NaT]
				})
>>
  name  toy       born
0    A  NaN        NaT
1    B    D 1940-04-25
2    C    E        NaT

>>>df.dropna()
  name toy       born
1    B   D 1940-04-25

>>>df.dropna(axis=1)
  name
0    A
1    B
2    C

>>>df.dropna(thresh=2)
  name toy       born
1    B   D 1940-04-25
2    C   E        NaT

>>>df.dropna(subset=['name', 'born'])
  name toy       born
1    B   D 1940-04-25
```

### .fillna()

**.fillna()：**填充DataFrame中的缺失值。

```python
'''
DataFrame.fillna( method=None,
				  limit,
				  axis,
				  inplace)
参数：
[1]method：取值有{'pad', 'ffill', 'backfill', 'bfill', None}，默认为None。
		  pad/ffill：用前一个非缺失值去填充该缺失值。
		  backfill/bfill：用下一个非缺失值填充该缺失值。
		  None：指定一个值去替换缺失值（缺省默认这种方式）。
[2]limit：限制填充个数。
[3]axis：修改填充方向。
'''

df = pd.DataFrame({"name": ['A', np.nan, 'C'],
                   "toy": [np.nan, 'D', 'E'],
                   "born": [pd.NaT, pd.Timestamp("1940-04-25"), pd.NaT]
				})
>>
  name  toy       born
0    A  NaN        NaT
1  NaN    D 1940-04-25
2    C    E        NaT
#用常数填充
>>>df.fillna(10)
  name toy                 born
0    A  10                   10
1   10   D  1940-04-25 00:00:00
2    C   E                   10
#用字典填充
>>>df.fillna({'name':10, 'toy':20, 'born':30})
  name toy                 born
0    A  20                   30
1   10   D  1940-04-25 00:00:00
2    C   E                   30
#用前一个非缺失值去填充该缺失值
>>>df.fillna(method='ffill')
  name  toy       born
0    A  NaN        NaT
1    A    D 1940-04-25
2    C    E 1940-04-25
#用下一个非缺失值填充该缺失值
>>>df.fillna(method='bfill')
  name toy       born
0    A   D 1940-04-25
1    C   D 1940-04-25
2    C   E        NaT
#与上面相同，默认axis=0
>>>df.fillna(method='bfill', axis=0)
  name toy       born
0    A   D 1940-04-25
1    C   D 1940-04-25
2    C   E        NaT
#后列的非残缺值填充前列的残缺值
>>>df.fillna(method='bfill', axis=1)
  name  toy       born
0    A  NaT        NaT
1    D    D 1940-04-25
2    C    E        NaT
```

### .quantile()

**.quantile()：**计算p分位数

```python
'''
DataFrame.quantile(
				   q=0.5,
				   axis=0,
				   interpolation='linear')
参数：
[1]q：计算分为点的概率，范围为[0,1]，默认为0.5（即二分位数）
[2]axis：0为'index'，1为'columns'，默认为0
[3]interpolation：插值方法，有{'linear', 'lower', 'higher', 'midpoint', 'nearest'}，默认'linear'。
注：
【1】当选中的分为点位于两个数数据点 i and j 之间时:
    linear: i + (j - i) * fraction.(fraction为计算获得的pos的小数部分)
    lower: i.
    higher: j.
    nearest: i or j whichever is nearest.
    midpoint: (i + j) / 2.
【2】 确定p分位数的位置：
	一、pos = (n+1)*p
	二、pos = 1+(n-1)*p
'''
df = pd.DataFrame(np.array([[1, 1], [2, 10], [3, 100],[4, 100]]), columns=['a', 'b'])
>>
   a    b
0  1    1
1  2   10
2  3  100
3  4  100
>>>df.quantile(.1)
a    1.3
b    3.7
Name: 0.1, dtype: float64
'''
计算a列
pos = 1 + (4 - 1)*0.1 = 1.3
fraction = 0.3

ret = 1 + (2 - 1) * 0.3 = 1.3

计算b列
pos = 1.3
ret = 1 + (10 - 1) * 0.3 = 3.7
'''
```

### .drop()

**.drop()：**使用该函数删除DataFrame指定行列。

```python
'''
df.drop(labels=None,
		axis=0,
		index=None,
		columns=None,
		level=None,
		inplace=False,
		error='raise'
		)
参数：
[1]labels：标签序列，会结合axis值，来删除带label标识的行或者列，如（labels='A', axis=1)表示A列。
[2]axis：0表示行，1 或 ‘columns’ 表示列。
[3]index：根据行标签 index=labels 从行中删除值，(labels, axis=0) 等价于 index=labels
[4]columns：根据列标签 columns=labels 从列中删除值，(labels, axis=1) 等价于 columns=labels
[5]inplace：True表示在原对象中操作，会清除被删除的数据；Flase表示不改变原对象，返回删除后的数据，非被删除的数据。
'''

```

### .T

**.T：**转置。

### .mean()

**.mean()：**对每行/列求均值，通过axis=0/1确定行列。

### .sort_values()

**.sort_values()：**沿着任意一个轴按值排序

```python
'''
df.sort_values(by, 
							 axis=0,
							 ascending=True,
							 inplace=False,
							 na_position='last'，
							 ignore_index=False,)
参数：
[1]by：指定行/列名或索引值（axis=0或‘index’)（axis=1或'columns'）
[2]axis：默认axis=0按列排序，axis=1按行排序
[3]ascending：选择升序（默认True），或者降序（False）
[4]inplace：默认为False，即不替换
[5]na_position：表示空值位置，默认为'last'即放最后，可取{'fast','last'}
[6]ignore_index：默认为False，为True时重建索引
'''
df = pd.DataFrame({
    'col1': ['A', 'A', 'B', np.nan, 'D', 'C'],
    'col2': [2, 1, 9, 8, 7, 4],
    'col3': [0, 1, 9, 4, 2, 3],
    'col4': ['a', 'B', 'c', 'D', 'e', 'F']
})
>>>df
  col1  col2  col3 col4
0    A     2     0    a
1    A     1     1    B
2    B     9     9    c
3  NaN     8     4    D
4    D     7     2    e
5    C     4     3    F
#按照col1排序
>>>df.sort_values(by=['col1'])
  col1  col2  col3 col4
0    A     2     0    a
1    A     1     1    B
2    B     9     9    c
5    C     4     3    F
4    D     7     2    e
3  NaN     8     4    D
#按多列排序
>>>df.sort_values(by=['col1', 'col2'])
  col1  col2  col3 col4
1    A     1     1    B
0    A     2     0    a
2    B     9     9    c
5    C     4     3    F
4    D     7     2    e
3  NaN     8     4    D
#降序排序
>>>df.sort_values(by='col1', ascending=False)
  col1  col2  col3 col4
4    D     7     2    e
5    C     4     3    F
2    B     9     9    c
0    A     2     0    a
1    A     1     1    B
3  NaN     8     4    D
#把NAs放在第一位
>>>df.sort_values(by='col1', ascending=False, na_position='first')
  col1  col2  col3 col4
3  NaN     8     4    D
4    D     7     2    e
5    C     4     3    F
2    B     9     9    c
0    A     2     0    a
1    A     1     1    B
```

### .rename()

.rename()：重命名任何索引，列或行。
【注】列的重命名也可以通过dataframe.columns = [#list]。但在上述情况下，自由度不高。即使必须更改一列，也必须传递完整的列列表。另外，上述方法不适用于索引标签。

```python
'''
.rename(mapper=None, index=None, columns=None, axis=None, copy=True, inplace=False, level=None, errors='ignore')
参数：
[1]mapper：映射器，为字典或函数。当为字典时，键表示旧名称，值表示新名称。
[2]axis：0表示行，1表示列。
[3]copy：为True，则复制基础数据。
[4]inplace：为True，则在原始DataFrame中进行修改。
[5]level：用于在数据帧具有多个级别索引的情况下指定级别。
返回：
具有新名称的DataFrame
'''
#案例
response = requests.post(url=url, data=data, headers=headers)
r_dframe = pd.DataFrame(response.json()['data']['tableChart'])
cl = r_dframe.loc[:, ['create_time', 'totalSum']]
>>>cl
            create_time  totalSum
0   2022-10-15 15:00:30    2000.0
1   2022-10-15 16:00:30    2000.0

cl.rename({'create_time':'time', 'totalSum':'cl'}, axis=1, inplace=True)
>>>cl
                   time      cl
0   2022-10-15 15:00:30  2000.0
1   2022-10-15 16:00:30  2000.0
2   2022-10-15 17:00:30  2050.0
```

### .groupby()

.groupby()：分组，返回Series对象。

### .set_index()

.set_index()：将dataframe中的一列设置为索引

```python
'''
.set_index(keys, drop=True, append=False, inplace=False, verify_integrity=False)
参数：
[1]keys：要设置为索引的列名，如有多个应该放在一个列表里
[2]drop：将设置为索引的列删除，默认为True
[3]append：是否将新的索引追加到原索引后（即是否保留原索引），默认为False
[4]inplace：是否在原DataFrame上修改，默认为False
[5]verify_integrity：是否检查索引有无重复，默认为False
'''
```



## datetime

### datetime

datetime：生成datetime对象
【注】可使用datetime的对象方法来获取它的date、time、year、month等对象

```python
from datetime import datetime, date, time
dt = datetime(2022, 1, 1, 21, 30, 28)
>>>dt
datetime.datetime(2022, 1, 1, 21, 30, 28)
>>>dt.time()
datetime.time(21, 30, 28)
>>>dt.date()
datetime.date(2022, 1, 1)
>>>print(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
2022 1 1 21 30 28
```

### .strftime()

.strftime()：将datetime数据按自己想要的格式转换为字符串。

```python
>>>dt.strftime('%m/%d/%Y %H:%M')
'01/01/2022 21:30'
>>>dt.strftime('%Y-%m-%d %H:%M:%S')
'2022-01-01 21:30:28'

#另一个案例
import pandas as pd
from datetime import datetime
date_ = pd.date_range('20190101', periods=8760, freq='H')
print(date_)
print(date_.strftime('%m%d %H'))
>>
DatetimeIndex(['2019-01-01 00:00:00', '2019-01-01 01:00:00',
               '2019-01-01 02:00:00', '2019-01-01 03:00:00',
               '2019-01-01 04:00:00'],
              dtype='datetime64[ns]', freq='H')
Index(['0101 00', '0101 01', '0101 02', '0101 03', '0101 04'], dtype='object')
```

### datetime.strptime()

datetime.strptime()：将字符串转换为datetime，需要传入字符串及其对应的格式

```python
from datetime import datetime
>>>datetime.strptime('20220101213028', '%Y%m%d%H%M%S')
datetime.datetime(2022, 1, 1, 21, 30, 28)
>>>datetime.strptime('2022-01-01 21:30:28', '%Y-%m-%d %H:%M:%S')
datetime.datetime(2022, 1, 1, 21, 30, 28)
```



## seaborn

### sns.

```python
import seaborn as sns
```

## statsmodels.api

### sm.

```python
import statsmodels.api as sm
```

## tensorflow

### tf.data.Dataset

#### tf.data.Dataset.from_tensor_slices()

**tf.data.Dataset.from_tensor_slices**()：把给定的元组、列表和张量等数据进行特征切片，返回一个`Dataset`。切片的范围是从最外层维度开始。如果有多个特征进行组合，那么一次切片是把每个组合的最外维度的数据切开，分成一组一组。

```python
'''
tf.data.Dataset.from_tensor_slice(tensor, name=None)
参数：
tensor：给定的张量沿它们的第一维进行切片。此操作保留输入张量的结构，删除每个张量的第一个维度并将其用作数据集维度。所有输入张量的第一个维度必须具有相同的大小。
'''
#切片1D张量生成标量tensor的对象
>>>dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
>>>dataset
<TensorSliceDataset shapes: (), types: tf.int32>
>>>list(dataset.as_numpy_iterator)
[1, 2, 3]

#切片2D张量生成1Dtensor的对象
>>>dataset = tf.data.Dataset.from_tensor_slices([[1, 2], [3, 4]])
>>>list(dataset.as_numpy_iterator())
[array([1, 2]), array([3, 4])]

#切片1D张量的元组生成由标量tensor组成的元组的对象
>>>dataset = tf.data.Dataset.from_tensor_slices(([1,2],[3,4],[5,6]))
>>>list(dataset.as_numpy_iterator())
[(1, 3, 5), (2, 4, 6)]

#用于dict
>>>dataset = tf.data.Dataset.from_tensor_slices({"a":[1,2],"b":[3,4]})
>>>list(dataset.as_numpy_iterator())
[{'a': 1, 'b': 3}, {'a': 2, 'b': 4}]

```

```python
#两个tensor组合成一个Dataset对象
features = tf.constant([[1, 3], [2, 1], [3, 3]])
labels = tf.constant(['A', 'B', 'A'])
dataset = tf.data.Dataset.from_tensor_slices((features, labels))
>>>list(dataset.as_numpy_iterator())
[(array([1, 3]), b'A'), (array([2, 1]), b'B'), (array([3, 3]), b'A')]

#该方式与上面创建的Dataset对象完全相同
features_dataset = tf.data.Dataset.from_tensor_slices(features)
labels_dataset = tf.data.Dataset.from_tensor_slices(labels)
dataset = tf.data.Dataset.zip((features_dataset, labels_dataset))

#更多例子
batched_features = tf.constant([[[1, 3], [2, 3]],
                                [[2, 1], [1, 2]],
                                [[3, 3], [3, 2]]], shape=(3, 2, 2))
batched_labels = tf.constant([['A', 'A'],
                             ['B', 'B'],
                             ['A', 'B']], shape=(3, 2, 1))
dataset = tf.data.Dataset.from_tensor_slices((batched_features, batched_labels))
>>>for element in dataset.as_numpy_iterator():
    print(element)
(array([[1, 3],
       [2, 3]]), array([[b'A'],
       [b'A']], dtype=object))
(array([[2, 1],
       [1, 2]]), array([[b'B'],
       [b'B']], dtype=object))
(array([[3, 3],
       [3, 2]]), array([[b'A'],
       [b'B']], dtype=object))
```

```python
features, labels = (np.random.sample((6,3)),
                    np.random.sample((6,1)))
>>>data = tf.data.Dataset.from_tensor_slices((features, labels))
<TensorSliceDataset shapes: ((3,), (1,)), types: (tf.float64, tf.float64)>
#即每3个特征对应1个标签
```

#### .as_numpy_iterator()

**.as_numpy_iterator()：**将`Dataset`的所有元素转化为`numpy`并组成为一个迭代器返回。

```python
#直接打印数据集元素
>>>dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
>>>for element in dataset:
    print(element)
tf.Tensor(1, shape=(), dtype=int32)
tf.Tensor(2, shape=(), dtype=int32)
tf.Tensor(3, shape=(), dtype=int32)

#转化后查看
>>>for element in dataset.as_numpy_iterator():
    print(element)
1
2
3

#as_numpy_iterator()将保留数据集元素的嵌套结构
>>>dataset1 = tf.data.Dataset.from_tensor_slices({'a':([1,2],[3,4]),'b':[5,6]})
>>>list(dataset1.as_numpy_iterator())
[{'a': (1, 3), 'b': 5}, {'a': (2, 4), 'b': 6}]
```

#### .take()

**.take()**：按顺序取元素构建Dataset

```python
dataset = tf.data.Dataset.from_tensor_slices([5, 4, 3, 2, 1])
>>>list(dataset.as_numpy_iterator())
[5, 4, 3, 2, 1]

t1 = dataset.take(1)#按顺序取dataset的一个元素构建新Dataset，不是随机的一个
>>>t1
<TakeDataset shapes: (), types: tf.int32>
>>>list(t1.as_numpy_iterator())
[5]

t2 = dataset.take(3)#按顺序取dataset的3个元素构建新Dataset
>>>t2
<TakeDataset shapes: (), types: tf.int32>
>>>list(t2.as_numpy_iterator())
[5, 4, 3]
```

#### .skip()

**.skip()：**按顺序跳过元素后构建Dataset

```python
t3 = dataset.skip(2)
>>>t3
<SkipDataset shapes: (), types: tf.int32>
>>>list(t1.as_numpy_iterator())
[3, 2, 1]

t4 = dataset.skip(2).take(2)
>>>list(t4.as_numpy_iterator())
[3, 2]
```



### tf.constant()

**tf.constant()：**创建常量tensor

```python
'''
tf.constant(value, dtype=None, shape=None, name='Const')
参数：
value：可以为 python list 或 numpy array 对象
dtype：若dtype不指定，则保持原value的type；若dtype指定，则将原value的type改为dtype。
shape：若shape指定，则value会重组。此时，标量会被广播来满足shape，不是标量不会被广播。
name：操作的名称
'''
#python list
>>>tf.constant([1, 2, 3])
<tf.Tensor: shape=(3,), dtype=int32, numpy=array([1, 2, 3])>
#numpy array
>>>a = np.array([[1, 2, 3], [4, 5, 6]])
>>>tf.constant(a)
<tf.Tensor: shape=(2, 3), dtype=int32, numpy=
array([[1, 2, 3],
       [4, 5, 6]])>

#指定dtype
>>>tf.constant([1, 2, 3], dtype=tf.float64)
<tf.Tensor: shape=(3,), dtype=float64, numpy=array([1., 2., 3.])>

#指定shape
>>>tf.constant(0, shape=(2, 3))
<tf.Tensor: shape=(2, 3), dtype=int32, numpy=
array([[0, 0, 0],
       [0, 0, 0]])>

>>>tf.constant([1,2,3,4,5,6], shape=(2,3))
<tf.Tensor: shape=(2, 3), dtype=int32, numpy=
array([[1, 2, 3],
       [4, 5, 6]])>
```

### tf.keras

#### tf.keras.layers

##### tf.keras.layers.LSTM()

**tf.keras.layers.LSTM()：**tensorflow中keras的LSTM层。

```python
tf.keras.layers.LSTM(
    units,
    activation="tanh",
    recurrent_activation="sigmoid",#为循环步施加的激活函数
    use_bias=True,
    kernel_initializer="glorot_uniform",#权重初始化
    recurrent_initializer="orthogonal",
    bias_initializer="zeros",#偏差初始化
    unit_forget_bias=True,
    kernel_regularizer=None,
    recurrent_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    recurrent_constraint=None,
    bias_constraint=None,
    dropout=0.0,
    recurrent_dropout=0.0,
    return_sequences=False,
    return_state=False,
    go_backwards=False,
    stateful=False,
    time_major=False,
    unroll=False,
    **kwargs
)
```

##### tf.keras.layers.Dense()

**tf.keras.layers.Dense()：**tensorflow中keras的全连接层Dense。

```python
tf.keras.layers.Dense(
    units,
    activation=None,#默认为线性linear:a(x)=x
    use_bias=True,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
)
```

##### tf.keras.layers.Conv1D()

**tf.keras.layers.Conv1D()：**一维卷积层（如时间卷积）

```python
tf.keras.layers.Conv1D(
    filters,
    kernel_size,
    strides=1,
    padding="valid",
    data_format="channels_last",
    dilation_rate=1,
    groups=1,
    activation=None,
    use_bias=True,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
)
'''
参数：
[1]kernel_size：一维卷积窗口的长度
[2]kernel_initializer：默认情况下，Keras使用具有均匀分布的Glorot初始化，即glorot_uniform
'''
```

#### tf.keras.optimizers

##### tf.keras.optimizers.Adam()

**tf.keras.optimizers.Adam()：**Adam优化器

```python
tf.keras.optimizers.Adam(
    learning_rate=0.001,#默认学习率为0.001
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name="Adam",
    **kwargs
)
```



## Pytorch

### tensor

**tensor：**张量，同数据和矩阵一样，是一种特殊的数据结构。在Pytorch中，神经网络的输入、输出以及网络的参数等数据均使用张量来进行描述。
					使用这个词汇的目的是为了表述统一，张量可以看作是向量、矩阵的自然推广，用张量来表示广泛的数据类型。
注：(1)可以将标量视为零阶张量，矢量/向量视为一阶张量，矩阵视为二阶张量。
				两个名词：标量张量与一维张量数组。
		(2)张量的阶数也称维度或轴。
		(3)在Python中，张量通常存储在Numpy数组。
		(4)

| 阶   | 数据实例                          | Python例子                                                   |
| ---- | --------------------------------- | ------------------------------------------------------------ |
| 0    | 纯量（只有大小）                  | s = 1                                                        |
| 1    | 向量（大小和方向）                | v = [1, 2, 3]                                                |
| 2    | 矩阵（数据表）                    | n = [[1, 2, 3], [4, 5, 6]]                                   |
| 3    | 3阶张量（数据立方体）（时间序列） | t = [ [ [1, 2], [3, 4], [4, 5] ], [ [6, 7], [8, 9], [10, 11] ] ] |
| 4    | 4阶张量（图像）                   |                                                              |
| 5    | 5阶张量（视频）                   |                                                              |
| n    | n阶张量                           |                                                              |

​		(5)例子
​			[1]时间序列数据
​			医学扫描——将脑电波信号编码成3D张量，由三个参数来描述：

```python
(time, frequency, channel)
```

​			如果有多个病人的脑电波扫描图，就形成了一个4D张量

```python
(sample_size, time, frequency, channel)
```

​			[2]图像数据
​			一个图像可以用三个参数描述：

```python
(width, height, color_depth)#高度、宽度和颜色深度
```

​			一张图片是3D张量，一个图片集则为4D张量，第4维是样本大小。

```python
(sample_size, width, height, color_depth)
```

​			[3]视频数据
​			一段5分钟（300秒），每秒15帧（总共4500帧）（一帧即一张画），1080pHD（1920 x 1080像素），颜色深度为3的视频，用4D张量存储表示为：

```python
(4500, 1920, 1080, 3)
```

​			有10段这样的视频，则5D张量表示为：

```python
(10, 4500, 1920, 1080, 3)
```

​		(6)通常，一个n阶张量，前(n-1)个参数用来描述一个样本，最后一个参数表示样本量，这是TensorFlow的数据组织形式，称为“channels_last”，表示为

```python
(sample_size, ...)#具有该shape的数组array，从里到外/从后向前，描述含义由小到大

'keras通过以下方式查看keras默认的数据组织形式'
from keras import backend as K
>>>K.image_data_format()
'channels_last'
```



#### tensor属性

```python
import torch
tensor = torch.rand(3,4)#生成从0~1随机采样的(3,4)的tensor
tensor.shape#(行数，列数)
tensor.dtype#数据类型
tensor.device#存储设备
>>
tensor([[0.5955, 0.4738, 0.6935],
        [0.5687, 0.2259, 0.9889]])
torch.Size([2, 3])
torch.float32
device(type='cpu')
```

#### .tensor()

**torch.tensor()：**将Python的list或序列转换成Tensor。

```python
torch.tensor([[1., -1.], [1., -1.]])
>>tensor([[ 1., -1.],
        [ 1., -1.]])
torch.tensor(np.array([[1, 2, 3], [4, 5, 6]]))
>>tensor([[1, 2, 3],
        [4, 5, 6]], dtype=torch.int32)
```

#### .numpy()

**t.numpy()：**将Tensor变量转换为ndarray变量，其中t是一个Tensor变量，可以是标量，也可以是向量，转换后dtype与Tensor的dtype一致。
注：要将Tensor转换为list的话，得先将Tensor转换为ndarray，再将ndarray转换为list。

```python
a = torch.tensor(1.)
b = torch.tensor([[1., 2.]])
aa = a.numpy()
bb = b.numpy()
>>
aa = array(1., dtype=float32)
bb = array([[1., 2.]], dtype=float32)
```

### torch.transpose()

**torch.transpose()：**交换矩阵的两个维度。

```python
'''
x为Tensor
torch.transpose(x, dim0, dim1)
x.transpose(dim0, dim1)
注：
torch.transpose(x, 1, 0)与torch.transpose(x, 0, 1)效果一样。
'''
x = torch.randn(2, 3)
>>
tensor([[-0.0673, -0.1480,  0.2352],
        [-0.7728,  1.0108, -0.6342]])
torch.transpose(x, 1, 0)
>>
tensor([[-0.0673, -0.7728],
        [-0.1480,  1.0108],
        [ 0.2352, -0.6342]])
torch.transpose(x, 0, 1)
>>
tensor([[-0.0673, -0.7728],
        [-0.1480,  1.0108],
        [ 0.2352, -0.6342]])
```

### torch.ones()

```python
tensor = torch.ones(4,4)
tensor[:,1] = 0#张量的索引和切片
>>
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])
```

### torch.cat()

**torch.cat()：**将一组张量按照指定的维度进行拼接。

```python
t1 = torch.cat([tensor,tensor], dim=1)#从外向内算维度，维度从0开始；dim=1即在1维进行拼接
>>
tensor([[1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.]])
```

### torch.utils.data.DataLoader

**torch.utils.data.DataLoader：**为类，表示可在数据集上迭代的Python，并支持：
(1)映射式和迭代式的数据集；
(2)自定义数据加载顺序；
(3)自动批次；
(4)单进程和多进程数据加载；
(5)自动内存固定。
**DataLoader的构造函数**如下：

```python
'''
DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None)
参数：
[1]dataset：指定用于加载数据的数据集对象。[Tips：PyTorch支持两种不同类型的数据集：映射式数据集 和 迭代式数据集]。
[2]batch_size：每批次要加载多少个样本。[Tips：若batch_size(默认1)不为None时，数据加载器将生成批处理的样本，而不是单个样本；若batch_size为None，数据加载器直接返回dataset对象的每个成员]。
[3]sampler：用该参数指定一个自定义的Sampler对象，该Sampler采样器定义从数据集中抽取样本的策略。若指定了Sampler，则shuffle必须为False。
[4]shuffle：为True时，数据每轮会被重新打乱(默认为False)。该参数将自动构建顺序采样或打乱的采样器。
'''

```

### torchvision.datasets

**torchvision.datasets：**收录了一些数据集，这些数据集都是`torch.utils.data.Dataset`的子类，它们都可以传递给`torch.utils.data.DataLoader`，进而通过`torch.multiprocessing`实现批数据的并行化加载。
<!--torchvision.datasets 的数据集的接口基本上很相近，至少包括两个公共的参数 transform 和 target_transform ，以便分别对输入和目标做变换。-->

#### MNIST数据集

```python
'''
class torchvision.datasets.MNIST(root, train=True, transform=None, target_transform=None, download=False)
参数：
[1]root(_string_)：数据集的根目录，其中存放processed/training.pt和processed/test.pt文件。
[2]train(bool,可选)：若为True，从training.pt创建数据集，否则从test.pt创建。
[3]download(bool,可选)：若为True，从互联网下载数据并放到root文件夹下；若root目录下已经存在数据，不会再次下载。
[4]transform(可被调用，可选)：一种函数或变换，输入PIL图片，返回变换后的数据，如transform.RandomCrop。
[5]target_transform(可被调用，可选)：一种函数或变换，输入目标，进行变换。
'''
```

### torchvision.transforms

**torchvision.transforms：**包含一些常用的图像变换，以及格式变换。

#### ToTensor

**torchvision.transform.ToTensor：**将`PIL Image`或`numpy.ndarray`转化成张量。

```python
class torchvision.transform.ToTensor()
```

### torchvision.utils

#### make_grid()

**torchvision.utils.make_grid()：**把图片排列成网格形状。

```python
'''
torchvision.utils.make_grid(tensor, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
参数：
[1]tensor(Tensor或list)：四维批(bathc)Tensor或list。若是Tensor，其形状应是(B*C*H*W)；若是list，元素应为相同大小的图片。
[2]nrow(int,可选)：最终展示的图片网格中每行摆放的图片数量。网格的长宽应是(B/nrow,nrow)。默认是8。
'''
```

## keras

### keras.models

<!--Keras有两种类型的模型：顺序模型（Sequential）和泛型模型（Model）-->
<!--Keras构建模型方式：使用顺序API构建模型；使用函数式API构建模型；使用子类API构建动态模型。-->
【Tips】keras的`model`的输入数据的形式为`(nb_samples, nb_timesteps, nb_dims)`或`(nb_samples, nb_dims)`。

#### Model

**Model：**Keras的泛型模型为`Model`，即广义的拥有输入和输出的模型，使用`Model`来初始化一个泛型模型。

```python
from keras.models import Model
from keras.layers import Input, Dense

a = Input(shape=(32, ))
b = Dense(32)(a)
model = Model(input=a, output=b)
#以上，模型以a为输入，以b为输出。同样，可以构造拥有多输入和多输出的模型
model = Model(input=[a1, a2], output=[b1, b2, b3])
```

##### Model属性

###### .layers

**model.layers：**组成模型图的各个层

###### .inputs

**model.inputs：**模型的输入张量列表

###### .outputs

model.outputs：模型的输出张量列表

##### Model方法

###### .compile()

**model.compile()：**本函数编译模型以供训练。
【Tips】如果只是载入模型并利用其predict，可以不用进行compile。在Keras中，compile主要完成损失函数和优化器的一些配置，是为训练服务的。predict会在内部进行符号函数的编译工作（通过调用_make_predict_function生成函数）。

```python
'''
.compile(self, optimizer, loss, metrics=[], loss_weights=None, sample_weight_mode=None)
参数：
[1]optimizer：优化器，为预定义优化器名或优化器对象
[2]loss：目标函数，为预定义损失函数名或一个目标函数
[3]metrics：列表，包含评估模型在训练和测试时的性能的指标，典型用法时metrics=['accuracy']。如果要在多输出模型中为不同的输出指定不同的指标，可向该参数传递一个字典，例如metrics={'output_a': 'accuracy'}
[4]sample_weight_mode：如果需要按时间步为样本赋权（2D权矩阵），将该值设为”temporal“。默认为”None“，代表按样本赋权（1D权）。如果模型有多个输出，可以向该参数传入指定sample_weight_mode的字典或列表.
'''
```

###### .fit()

**model.fit**()：本函数用以训练模型，将模型训练`nb_epoch`轮。
【Tips】`fit`函数返回一个`History`对象，其`History.history`属性记录了损失函数和其他指标的数值随`epoch`变化的情况，如果有验证集的话，也包含了验证集的这些指标变化情况。

```python
'''
.fit(self, 
		x, 
		y, 
		batch_size=32, 
		nb_epoch=10, 
		verbose=1, 
		callbacks=[], 
		validation_split=0.0,
		validation_data=None,
		shuffle=True,
		class_weight=None,
		sample_weight=None
		)
参数：
[1]x：输入数据。如果模型只有一个输入，那么x的类型是numpy array，如果模型有多个输入，那么x的类型应当为list，list的元素是对应于各个输入的numpy array。如果模型的每个输入都有名字，则可以传入一个字典，将输入名与其输入数据对应起来。
[2]y：标签，numpy array。如果模型有多个输出，可以传入一个numpy array的list。如果模型的输出拥有名字，则可以传入一个字典，将输出名与其标签对应起来。
[3]batch_size：整数，指定进行梯度下降时每个batch包含的样本数。训练时一个batch的样本会被计算一次梯度下降，使目标函数优化一步。
[4]nb_epoch：整数，训练的轮数，训练数据将会被遍历nb_epoch次。Keras中”nb“开头的变量均为”number of“的意思。
[5]verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录。
[6]callbacks：list，其中的元素是keras.callbacks.Callback的对象。这个list中的回调函数将会在训练过程中的适当时机被调用。
[7]validation_split：0~1的浮点数，用来指定训练集的一定比例数据作为验证集。验证集将不参与训练，并在每个epoch结束后测试模型的指标，如损失函数、精确度等。
[8]validation_data：形式为(x, y)的tuple，是指定的验证集。此参数将覆盖validation_split。
[9]shuffle：布尔值或字符串，一般为布尔值，表示是否在训练过程中随机打乱输入样本的顺序。若为真，训练的数据就会被随机洗乱，训练数据会在每个epoch的训练中都被重新洗乱一次，验证集的数据不会被洗乱。若为字符串"batch"，则是用来处理HDF5数据的特殊情况，它将在batch内部将数据打乱。不设置时默认为真。
[10]class_weight：字典，将不同的类别映射为不同的权值，该参数用来在训练过程中调整损失函数（只能用于训练）。
[11]sample_weight：权值的numpy array，用于在训练时调整损失函数（仅用于训练）。可以传递一个1D与样本等长的向量用于对样本进行1对1的加权，或者在面对时序数据时，传递一个形式为(samples, sequence_length)的矩阵来为每个时间步上的样本赋不同的权。这种情况下请确定在编译模式时添加了sample_weight_mode="temporal"。
[12]initial_epoch：从该参数指定的epoch开始训练，在继续之前的训练时有用。
'''
```

<!--如果没有特殊说明，以下函数的参数均保持与fit的同名参数相同的含义。-->
<!--如果没有特殊说明，以下函数的verbose参数（如果有）均只能取0或1。-->
###### .evaluate()

**model.evaluate()：**本函数按batch计算在某些输入数据上模型的误差。本函数返回一个测试误差的标量值（如果模型没有其他评价指标），或一个标量的list（如果模型还有其他的评价指标）。model.metrics_names将给出list中各个值的含义。

```python
'''
model.evaluate(self,
				x,
				y,
				batch_size=32,
				verbose=1,
				sample_weight=None
				)
参数：
x, y, batch_size, sample_weight：含义同fit的同名参数
verbose：含义同fit的同名参数，但只能取0或1
'''
```

###### .predict()

**model.predict()：**本函数按batch获得输入数据对应的输出。
注：函数的返回值是预测值的`numpy array`。

```python
'''
model.predict(self,
				x,
				y,
				batch_size=32,
				verbose=0
				)
'''
```



#### Sequential

顺序API/顺序模型Sequential适用于简单的层堆栈，其中每一层恰好有一个输入张量和输出张量。
<!--Sequential的属性和方法同Model-->

```python
'''
作为Sequential模型的第一层，需要指定输入维度。可以为 input_shape=(16,)或者 input_dim=16，这两者等价。
model = Sequential()
model.add(Dense(32, input_shape(16,)))
现在的模型就会以尺寸为(*, 16)的数组作为输入，
其输出数组的尺寸为(*, 32)。
在第一层后，就不再需要指定输入尺寸：
model.add(Dense(32))
'''
from keras.models import Sequential
model = Sequential()

from keras.layers import Dense#使用.add()来堆叠模型
model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))

#在完成模型的构建后，可以使用.compile()来配置学习过程
model.compile(loss='categorical_crossentropy',#损失函数
             optimizer='sgd',#优化器
             metrics=['accuracy']#评价指标
             )

#批量地在训练数据上进行迭代，x_train和y_train是Numpy数组
model.fit(x_train,
          y_train, 
          epochs=5, 
          batch_size=32,
          validation_split=0,#验证集比例
          verbose=0#静默模式；如果=1表示日志模式，输出每轮训练的结果
         )#返回损失函数loss值、评价指标metrics值

#手动地将批次的数据提供给模型
model.train_on_batch(x_batch, y_batch)

#评估模型性能
loss_and_metrics = model.evaluate(x_test, 
                                  y_test, 
                                  batch_size=128
                                 )#返回损失函数loss值、评价指标metrics值

#对新的数据进行预测
classes = model.predict(x_test, batch_size=128)

#输出model的详情
model.summary()
```


#### load_model

**keras.models.load_model()：**使用该函数来重新实例化模型，如果文件中存储了训练配置，该函数还会同时完成模型的编译。

```python
from keras.models import load_model
model.save('my_model.h5')#创建一个HDF5文件
del model#清除变量model对现存模型的引用

#重新实例化模型
#现model引用的模型与前一个模型完全相同
model = load_model('my_model.h5')
```

### keras.layers.core

<!--常用层对应于core模块，core内部定义了一系列常用的网络层，包括全连接、激活层等。-->

#### Dense

**keras.layers.core.Dense()：**常用的全连接层，或密集层。

```python
'''
keras.layers.core.Dense(
			output_dim, 
			init='glorot_uniform',
			activation='linear', 
			weight=None, 
			W_regularizer=None, 
			b_regularizer=None, 
			activity_regularizer=None, 
			W_constraint=None, 
			b_constraint=None, 
			bias=True, 
			input_dim=None
			)
参数：
[1]output_dim：大于0的整数，代表该层的输出维度。模型中非首层的全连接层其输入维度可以自动推断，因此非首层的全连接定义时不需要指定输入维度。
[2]activation：激活函数，为预定义的激活函数名。如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）
[3]input_dim：整数，输入数据的维度。当Dense层作为网络的第一层时，必须指定该参数或input_shape参数。
'''

#Sequential model的首层
model = Sequential()
model.add(Dense(32, input_dim=16))
#该model的输入数组的shape为(*, 16)，输出数组的shape为(*, 32)

#与上面model相同
model = Sequential()
model.add(Dense(32, input_shape=(16,)))

#若非首层，不需要特别指定输入维度
model.add(Dense(32))

'全连接网络'
'''
几个概念：
[1]层对象接受张量为参数，返回一个张量。
[2]输入是张量，输出也是张量的一个框架就是一个模型，通过Model定义。
[3]这样的模型可以被像Keras的Sequential一样被训练。
'''
from keras.layers import Input, Dense
from keras.models import Model

#返回一个张量tensor
inputs = Input(shape=(784,))

#层layers可以被tensor调用，并返回tensor
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

#创建了一个model，由一个输入层和三个全连接层（Dense layers）组成
model = Model(inputs=inputs, outputs predictions)
model.compile(optimizer='rmsprop',
             loss='categorical_crossentropy',
             metrics=['accuracy'])
model.fit(data, labels)
```

#### Activation

**keras.layers.core.Activation()：**激活层对一个层的输出施加激活函数。
【Tips】激活函数可以通过设置单独的`激活层`实现，也可以在构造层对象通过传递`activation`参数实现。

```python
'''
keras.layers.core.Activation(activation)
参数：
activation：将要使用的激活函数，为预定义激活函数名或一个Tensorflow函数。
输入shape：任意，当使用激活层作为第一层时，要制定input_shape。
输出shape：与输入shape相同。
'''

from keras.layers.core import Activation, Dense
model.add(Dense(64))
model.add(Activation('tanh'))

#以下建模跟上面完全相同
model.add(Dense(64, activation='tanh'))
```

#### Dropout

**keras.layers.core.Dropout()：**为输入数据施加Dropout。Dropout将在训练过程中每次更新参数时按一定概率（rate）随机断开输入神经元，Dropout层用于防止过拟合。

```python
'''
keras.layers.core.Dropout(rate, noise_shape=None, seed=None)
参数：
rate：0~1的浮点数，控制需要断开的神经元的比例
noise_shape：整数张量，为将要应用在输入上的二值Dropout mask的shape，例如你的输入为(batch_size, timesteps, features)，并且你希望在各个时间步上的Dropout mask都相同，则可传入noise_shape=(batch_size, 1, features)。
seed：整数，使用的随机数种子
'''
```

#### Flatten

**keras.layers.core.Flatten()：**Flatten层用来将输入“压平”，即把**多维的输入一维化**，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。

```python
model = Sequential()
model.add(Convolution2D(64, 3, 3,
                       border_mode='same',
                       input_shape=(3, 32, 32)))
#当前：model.output_shape == (None, 64, 32, 32)
model.add(Flatten())
#当前：model.output_shape == (None, 65536)
```

#### Reshape

**keras.layers.core.Reshape()：**Reshape层用来将输入shape转换为特定的shapeK。

```python
'''
keras.layers.core.Reshape(target_shape)
参数：
[1]target_shape：目标shape，为整数的tuple，不包含样本数目的维度（batch大小）
输入shape：
任意，但输入的shape必须固定。当使用该层为模型首层时，需要指定input_shape参数
输出shape：
(batch_size, ) + target_shape 
'''
#作为Sequential model首层
model = Sequential()
model.add(Reshape((3, 4), input_shape=(12,)))
#当前：model.output_shape == (None, 3, 4)
#注意：'None'为batch大小

#作为Sequential model的中间层
model.add(Reshape((6, 2)))
#当前：model.output_shape == (None, 6, 2)

#也可以使用'-1'作为维度，以自动计算
model.add(Reshape((-1, 2, 2)))
#当前：model.output_shape == (None, 3, 2, 2)

```

#### Permute

**keras.layers.core.Permute()：**Permute层将输入的维度按照给定模式进行重排，例如，当需要将RNN和CNN网络连接时，可能会用到该层。

```python
'''
keras.layers.core.Permute(dims)
参数：
dims：整数tuple，指定重排的模式，不包含样本数的维度。重排模式的下标从1开始。例如(2, 1)代表将输入的第二个维度重排到输出的第一个维度，而将输入的第一个维度重排到第二个维度。
'''
model = Sequential()
model.add(Permute((2, 1), input_shape=(10, 64)))
#当前：model.output_shape == (None, 64, 10)
#注意：'None'是batch大小
```

#### Lambda

**keras.layers.core.Lambda()：**本函数用以对上一层的输出施以任何Theano/TensorFlow表达式

```python
'''
keras.layers.core.Lambda(function, output_shape=None, mask=None, arguments=None)
参数：
function：要实现的函数，该函数仅接受一个变量，即上一层的输出
output_shape：函数应该返回的值的shape，可以是一个tuple，也可以是一个根据输入shape计算输出shape的函数
输出shape：
由output_shape参数指定的输出shape，当使用tensorflow时可自动推断
'''
model.add(Lambda(lambda x: x ** 2))
### keras.layers.normalization
```

#### BatchNormalization

**keras.layers.normalization.BatchNormalization()：**该层在每个batch上将前一层的激活值重新规范化，即使得其输出数据的均值接近0，其标准差接近1。
**[Tips]BN层的作用**

- 加速收敛
- 控制过拟合，可以少用或不用Dropout和正则
- 降低网络对初始化权重不敏感
- 允许使用较大的学习率

### keras.layers.recurrent

#### Recurrent

**keras.layers.recurrent.Recurrent()：**这是循环层的抽象类，请不要在模型中直接应用该层（因为它是抽象类，无法实例化任何对象）。请使用它的子类LSTM，GRU或SimpleRNN。所有的循环层（LSTM，GRU，SimpleRNN）都继承本层，因此下面的参数可以在任何循环层中使用。

```python
'''
keras.layers.recurrent.Recurrent(return_sequences=False, input_dim, input_length)
参数：
[1]return_sequences：布尔值，默认False，控制返回类型。若为True则返回整个序列，否则仅返回输出序列的最后一个输出。
[2]input_dim：输入维度，当使用该层为模型首层时，应指定该值（或等价的指定input_shape）
[3]input_length：当输入序列的长度固定时，该参数为输入序列的长度。当需要在该层后连接Flatten层，然后又要连接Dense层时，需要指定该参数，否则全连接的输出无法计算出来。注意，如果循环层不是网络的第一层，你需要在网络的第一层中指定序列的长度（通过input_shape指定）。
输入shape：
形如(samples, timesteps, input_dim)的3D张量
输出shape：
如果return_sequences=True：返回形如(samples, timesteps, output_dim)的3D张量；否则，返回形如(samples, output_dim)的2D张量。
'''

#当为Sequential model的首层时
model = Sequential()
model.add(LSTM(32, input_shape=(10, 64)))
#当前：model.output_shape == (None, 32)
#注意：'None'为批大小

#以下建模跟上面完全相同
model = Sequential()
model.add(LSTM(32, input_dim=64, input_length=10))

#对于之后的层layers，不需要特别指定input size
model.add(LSTM(16))

#当堆叠循环层，必须使任何传递给下一循环层的循环层 的参数"return_sequences=True"
#注意：需要特别指定首层的input size
model = Sequential()
model.add(LSTM(64, input_dim=64, input_length=10, return_sequences=True))
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(10))
```

#### LSTM

**keras.layers.recurrent.LSTM()：**Keras长短期记忆模型。

```python
'''
keras.layers.recurrent.LSTM(
				units, 
				activation='tanh', 
				recurrent_activation='hard_sigmoid')
参数：
units：输出维度
activation：激活函数，为预定义的激活函数名
recurrent_activation：为循环步施加的激活函数
'''
```

#### SimpleRNN

**keras.layers.SimpleRNN()：**简单RNN。

```python
'''
(1)不需要指定输入序列的长度，因为循环神经网络可以处理任意数量的时间步长，这就是将第一个输入的维度设置为None的原因。
(2)默认情况下，Keras中的循环层仅返回最终输出。要使它们每个时间步长返回一个输出，必须设置 return_sequences=True。
(3)
'''
model_1 = keras.models.Sequential([
		keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
		keras.layers.SimpleRNN(20, return_sequences=True),
		keras.layers.SimpleRNN(1)
])
model_2 = keras.models.Sequential([
		keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
		keras.layers.SimpleRNN(20, return_sequences=True),
		keras.layers.Dense(1)
])
```

### keras.layers.convolutional

#### Convolution1D()

**keras.layers.convolutional.Convolution1D()：**一维卷积层，用在一维输入信号上进行领域滤波。当使用该层作为首层时，需要提供关键字参数input_dim或input_shape。例如input_dim=128表示长为128的向量序列输入，而input_shape(10, 128)代表10个由长为128的向量组成的向量序列。

```python
keras.layers.convolutional.Convolution1D(
  		nb_filter, 
  		filter_length, 
  		init='uniform', 
  		activation='linear', 
  		weights=None, 
  		border_mode='valid', 
  		subsample_length=1, 
  		W_regularizer=None, 
  		b_regularizer=None, 
  		activity_regularizer=None, 
  		W_constraint=None, 
  		b_constraint=None, 
  		bias=True, 
  		input_dim=None, 
  		input_length=None)
'''
参数：
[1]nb_filter：卷积核的数目（即输出的维度）
[2]filter_length：卷积核的空域或时域长度，又为kernel_size，一维卷积窗口的长度
[3]activation：激活函数，为预定义的激活函数名，或逐元素的Theano函数。如果不指定该函数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）
'''
```

### keras.callbacks

**回调函数Callbacks：**该函数是一组在训练的特定阶段被调用的函数集，可以使用回调函数来观察训练过程中网络内部的状态和统计信息。通过传递回调函数列表到模型的`.fit()`中，即可在给定的训练阶段调用该函数集中的函数。
【Tips】虽然称回调“函数”，但事实上Keras的回调函数是一个类，回调函数只是习惯性称呼。

#### keras.callbacks.History()

**keras.callbacks.History**()：该回调函数在Keras模型上会被自动调用，History对象即为`fit`方法的返回值。

#### keras.callbacks.ModelCheckpoint()

**keras.callbacks.ModelCheckpoint()**：该回调函数将在每个`epoch`后保存模型到`filepath`。
【解析】ModelCheckpoint回调会定期保存模型的检查点，默认情况下，在每个轮次结束时。
【Tips】`filepath`可以是格式化的字符串，里面的占位符将会被`epoch`值和传入`on_epoch_end`的`logs`关键字所填入。例如，`filepath`若为`weights.{ epoch : 02d - { val_loss : .2f}}.hdf5`，则会生成对应`epoch`和验证集`loss`的多个文件。

```python
'''
keras.callbacks.ModelCheckpoint(filepath, 
								monitor='val_loss', 
								verbose=0, 
								save_best_only=False,
								save_weights_only=False,
								mode='auto',
								period=1
								)
参数：
[1]filepath：字符串，保存模型的路径
[2]monitor：需要监视的值
[3]verbose：信息展示模式，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
[4]save_best_only：当设置为True时，将只保存在验证集上性能最好的模型
[5]mode：'auto'，'min'，'max'之一，在 save_best_only=True 时决定性能最佳模型的评判准则。例如，当监测值为val_acc时，模式应为max，当检测值为val_loss时，模式应为min。在auto模式下，评价准则由被监测值的名字自动推断。
[6]save_weights_only：若设置为True，则只保存模型权重，否则将保存整个模型（包括模型结构，配置信息等）
[7]period：CheckPoint之间的间隔的epoch数
'''
```



#### keras.callbacks.EarlyStopping()

**keras.callbacks.EarlyStopping**()：当监测值不再改善时，该回调函数将中止训练。
【解析】EarlyStopping回调会在多个轮次（即patience次）的验证集上没有任何进展时，中断训练，并且可以选择回滚到最佳模型。

```python
'''
keras.callbacks.EarlyStopping(
				monitor='val_loss', 
				patience=0, 
				verbose=0,
				mode='auto',
				min_delta=0
				)
参数：
[1]monitor：监控的变量
[2]patience：在监督指标没有提升的情况下，epochs等待轮数，等待大于该值监督指标始终没有提升，则提前停止训练。
		当early stop被激活（如发现loss相比上一个epoch训练没有下降），则经过patience个epoch后停止训练。表示能容忍监督指标几次不按照优化过程迭代，如monitor=val_loss，patience=3，表示如果连续3次val_loss不减小，就停止迭代。
[3]verbose：信息展示模式
[4]mode：'auto'，'min'，'max'之一，在min模式下，如果检测值停止下降则中止训练。在max模式下，当检测值不再上升则停止训练。
[5]min_delta：monitor的最小变化，如果绝对值小于min_delta，则可以看作对结果没有improvement，默认为0。
'''
```

#### keras.callbacks.TensorBoard()

**keras.callbacks.TensorBoard**()：该回调函数是一个可视化的展示器。TensorBoard是TensorFlow提供的可视化工具，该回调函数将日志信息写入TensorBoard，使得你可以动态的观察训练和测试指标的图像以及不同层的激活值直方图。

```python
'''
keras.callbacks.TensorBoard(
				log_dir='./logs',
				histogram_freq=0
				)
参数：
log_dir：保存日志文件的地址，该文件将被TensorBoard解析以用于可视化
histogram_freq：计算各个层激活值直方图的概率（每多少个epoch计算一次），如果设置为0则不计算
'''
```



### preprocessing

#### preprocessing.timeseries_dataset_from_array()

**preprocessing.timeseries_dataset_from_array()：**在以array数组形式提供的时间序列上创建滑动窗口数据集。生成时间序列数据集，即由data生成sequence，每个sequence对应一个target标签（要预测的值）。

```python
'''
tensorflow.keras.preprocessing.timeseries_dataset_from_array(
		data,
    targets,
    sequence_length,
    sequence_stride=1,
    sampling_rate=1,
    batch_size=128,
    shuffle=False,
    seed=None,
    start_index=None,
    end_index=None,
)
参数：
[1]data：表示 x 数据，x 是以timestep记录的连续数据点的numpy数组或eager tensor，。
[2]targets：表示 y 标签，对应于在data中timestep的target，如果不处理标签只处理数据，则 targets=None。
[3]sequence_length：输出序列(sequence)的长度（是timestep的数量）。
[4]sequence_stride：连续输出序列的周期。当 stride 为 s，输出索引为 data[i], data[i+s], data[i+2*s]，etc。默认为1. 
[5]sampling_rate：在序列sequence中连续单一timesteps间的周期。当 rate 为 r，timesteps为data[i]，data[i+r]，...data[i+sequence_length]用于创建样本序列。默认为1.
[6]batch_size：可能除了最后一批外，每批次的时间序列的样本数。如果不设置，则不会对数据进行批处理(数据集将产生单个样本)。
[7]shuffle：是否随机输出样本，或者按时间顺序输出。shuffle为真，划分完时间序列后进行打乱，之后再分批。
[8]seed：可选的整数，用于shuffle的随机种子。
[9]start_index：可选的整数，start_index前的数据点不用于输出序列(不包含start_index)，常用于保留部分数据用于测试或验证。
[10]end_index：可选的整数，end_index后的数据点不用于输出序列(不包含start_index)。
注：
（1）return为tf.data.Dataset，如果有targets，dataset为元组(batch_of_sequences, batch_of_targets)，如果没有targets，dataset仅为batch_of_sequences。
（2）在Keras_LSTM_weather prediction例子中，由前720个时间戳的数据，预测720+72时间戳处的标签targets，该sequence长度在采样频率为1时就是720。自变量x与因变量y是相差792对应。
'''
#案例1
#数据索引为[0, 1, ..., 99]，当sequence_length=10，sampling_rate=2，sequence_stride=3，shuffle=False，则该数据集将产生由以下索引组成的批次序列。
First sequence:  [0  2  4  6  8 10 12 14 16 18]
Second sequence: [3  5  7  9 11 13 15 17 19 21]
Third sequence:  [6  8 10 12 14 16 18 20 22 24]
...
Last sequence:   [78 80 82 84 86 88 90 92 94 96]#最后三个数据点被丢弃，因为无法生成包含它们的完整序列（下一个序列将从索引81开始，因此最后一步将超过99
注意：[0, 1, ..., 99]的99表示滑动窗口的“尾”截至99，而不是滑动窗口的“头”截至99。
```

```python
#案例2，data和targets分开处理
import numpy as np
import pandas as pd
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing

data = np.array([i for i in range(20)])#[0,1,...,19]
targets = np.array([i for i in range(11)])#[0,1,...,10]

data_timeseries = keras.preprocessing.timeseries_dataset_from_array(
					data=data,
					targets=None,
					sequence_length=10
					)
targets_timeseries = keras.preprocessing.timeseries_dataset_from_array(
					data = targets,
					targets = None,
					sequence_length = 1
					)
>>>list(data_timeseries)
[<tf.Tensor: shape=(11, 10), dtype=int32, numpy=
array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
       [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10],
       [ 2,  3,  4,  5,  6,  7,  8,  9, 10, 11],
       [ 3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
       [ 4,  5,  6,  7,  8,  9, 10, 11, 12, 13],
       [ 5,  6,  7,  8,  9, 10, 11, 12, 13, 14],
       [ 6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
       [ 7,  8,  9, 10, 11, 12, 13, 14, 15, 16],
       [ 8,  9, 10, 11, 12, 13, 14, 15, 16, 17],
       [ 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
       [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]])>]
>>>list(targets_timeseries)
array([[ 0],
       [ 1],
       [ 2],
       [ 3],
       [ 4],
       [ 5],
       [ 6],
       [ 7],
       [ 8],
       [ 9],
       [10]])>]
```

```python
#案例3，data和targets合并处理
data = np.array([i for i in range(20)])#[0,1,...,19]
targets = np.array([i for i in range(11)])#[0,1,...,10]

targets_app = np.zeros(data.size - targets.size, dtype=int)#用0填补targets
targets = np.append(targets, targets_app).reshape(data.shape)
>>>targets
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10,  0,  0,  0,  0,  0,  0,
        0,  0,  0])

timeseries = keras.preprocessing.timeseries_dataset_from_array(
				data = data,
				targets = targets,
				sequence_length = 10
				)
>>>list(timeseries)
[(<tf.Tensor: shape=(11, 10), dtype=int32, numpy=
array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
       [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10],
       [ 2,  3,  4,  5,  6,  7,  8,  9, 10, 11],
       [ 3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
       [ 4,  5,  6,  7,  8,  9, 10, 11, 12, 13],
       [ 5,  6,  7,  8,  9, 10, 11, 12, 13, 14],
       [ 6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
       [ 7,  8,  9, 10, 11, 12, 13, 14, 15, 16],
       [ 8,  9, 10, 11, 12, 13, 14, 15, 16, 17],
       [ 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
       [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]])>, 
  	<tf.Tensor: shape=(11,), dtype=int32, numpy=
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])>)]
'注：targets的shape必须同data的shape一致，即使targets不需要那么多。'
```

### keras.initializers

**keras.initializers**：Initializers是所有初始化方法的父类，不能直接使用，如果想要定义自己的初始化方法，请继承此类。
**初始化方法**：

- 定义了对Keras层设置初始化权重的方法。
- 不同的层可能使用不同的关键字来传递初始化方法，一般来说指定初始化方法的关键字是`kernel_initializer`和`bias_initializer`，以及`init`，例如：

```python
model.add(Dense(64,
               	kernel_initializer='random_uniform',
               	bias_initializer='zeros'))

model.add(Dense(64, init='uniform'))
```

- 一个初始化器可以由字符串指定（必须使下面的预定义初始化器之一），或一个`callable`的函数，例如：

```python
from keras import initializers

model.add(Dense(64, kernel_initializer=initializers.random_normal(stddev=0.01)))
#同样作用：使用默认字符串
model.add(Dense(64, kernel_initializer='random_normal'))
```

#### Zeros

**keras.initializers.Zeros**()：全零初始化

#### Ones

**keras.initializers.Ones**()：全1初始化

#### Constant

**keras.initializers.Constant**()：初始化为固定值value

```python
keras.initializers.Constant(value=0)
```

#### RandomNormal

**keras.initializers.RandomNormal**()：正态分布初始化

```python
'''
keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
参数：
mean：均值
stddev：标准差
seed：随机数种子
'''
```

#### RandomUniform

**keras.initializers.RandomUniform**()：均匀分布初始化

```python
'''
keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
参数：
minval：均匀分布下边界
maxval：均匀分布上边界
seed：随机数种子
'''
```

### Objectives

**Objectives**：目标函数，或称损失函数，是编译一个模型必须的两个参数之一：

```python
model.compile(loss='mean_squared_error', optimizer='sgd')
```

可以通过传递预定义目标函数名字指定目标函数，也可以传递一个Theano/TensorFlow的符号函数作为目标函数，该函数对每个数据点应该只返回一个标量值，并以下列两个参数为参数：

- y_true：真是的数据标签，Theano/TensorFlow张量
- y_pred：预测值，与y_true相同shape的Theano/TensorFlow张量

真实的优化目标函数是在各个数据点得到的损失函数之和的均值
**可用的目标函数**

- mean_squared_errormse
- mean_absolute_errormae
- mean_absolute_percentage_errormape
- hinge：L1损失
- squared_hinge：L2损失，squared_hinge是hinge的平方

### keras.optimizers

**optimizers：**优化器，是编译一个模型必须的两个参数之一。

```python
model = Sequential()
model.add(Dense(64, init='uniform', input_dim=10))
model.add(Activation('tanh'))
model.add(Activation('softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)
```

可以在调用`model.compile()`之前初始化一个优化器对象，然后传入该函数（如上所示），也可以在调用`model.compile()`时传递一个预定义优化器名。在后者情形下，优化器的参数将使用默认值。

```python
#pass optimizer by name: default parameters will be used
model.compile(loss='mean_squared_error', optimizer='sgd')
```

**所有优化器都可用的参数：**

```python
'clipnorm和clipvalue是所有优化器都可以使用的参数，用于对梯度进行裁剪'

'clipnorm=1.表示所有参数梯度被裁剪为最大范数为1.'
sgd = SGD(lr=0.01, clipnorm=1.)

'clipvalue=0.5表示所有参数梯度被裁剪为最大值为0.5，最小值为-0.5'
sgd = SGD(lr=0.01, clipvalue=0.5)
```

#### SGD

**keras.optimizers.SGD()：**随机梯度下降法，支持动量参数，支持学习衰减率，支持Nesterov动量

```python
'''
keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
参数：
[1]lr：大于0的浮点数，学习率
[2]momentum：大于0的浮点数，动量参数
[3]decay：大于0的浮点数，每次更新后的学习率衰减值
[4]nesterov：布尔值，确定是否使用Nesterov动量
'''
```

#### RMSprop

**keras.optimizers.RMSprop()：**该优化器通常是面对递归神经网络时的一个良好选择

```python
'''
keras.optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=1e-06)
【Tips】除学习率可调整外，建议保持优化器的其他默认参数不变
参数：
[1]rho：大于0的浮点数
[2]epsilon：大于0的浮点数，防止除0错误
'''
```

#### Adam

**keras.optimizers.Adam()：**一种随机优化方法

```python
'''
keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
该优化器的默认值来源于参考文献<Adam-A Method for Stochastic Optimization>
参数：
[1]lr：大或等于0的浮点数，学习率
[2]beta_1/beta_2：浮点数，0<beta<1，通常很接近1
[3]epsilon：大或等于0的浮点数，防止除0错误
'''
```

### keras.regularizers

**regularizers：**正则项，在优化过程中层的参数或层的激活值添加惩罚项，这些惩罚项将与损失函数一起作为网络的最终优化目标。
【Tips】

- 惩罚项基于层进行惩罚，目前惩罚项的接口与层有关，但Dense，TimeDistributedDense，MaxoutDense，Convolution1D，Convolution2D，Convolution3D具有共同的接口。
- 正则项通常用于对模型的训练施加某种约束，L1正则项即L1范数约束，该约束会使被约束矩阵/向量更稀疏。L2正则项即L2范数约束，该约束会使被约束的矩阵/向量更平滑，因为它对脉冲型的值有很大的惩罚。
- 这些层有三个关键字参数以施加正则项：

`W_regularizer`：施加在权重上的正则项，为`WeightRegularizer`对象

`b_regularizer`：施加在偏置向量上的正则项，为`WeightRegularizer`对象

`activity_regularizer`：施加在输出上的正则项，为`ActivityRegularizer`对象

```python
from keras.regularizers import l2, activity_l2
model.add(Dense(64, 
                input_dim=64, 
                W_regularizer=l2(0.01), 
                activity_regularize=activity_l2(0.01)))
```

### keras.utils.visualize_util

**keras.utils.visualize_util：**该模块提供了画出Keras模型的函数（利用graphviz）
该函数将画出模型结构图，并保存为图片：

```python
from keras.utils.visualize_util import plot
plot(model, to_file='model.png')
'''
plot接收两个可选参数：
show_shapes：指定是否显示输出数据的形状，默认为False
show_layers_names：指定是否显示层名称，默认为True
'''
```

### .save()

**.save()：**将Keras模型或权重保存在一个HDF5文件中，该文件包含：

- 模型的结构，以便重构该模型

- 模型的权重

- 训练配置（损失函数，优化器等）

- 优化器的状态，以便于从上一次训练中断的地方开始

```python
'''
model.save(filepath)
参数：
model：模型变量
filepath：路径变量
'''
```

### .fit()

**.fit()：**训练模型，返回一个History对象，其中含有的history属性包含了训练过程中损失函数的值以及其他度量指标随epoch变化的情况，若有验证集的话，也包含了验证集的这些指标变化情况。

```python
'''
model.fit(
		x,
		y,
		batch_size=32,
		epochs=10,
		verbose=1,
		callbacks=None,
		validation_split=0.0,
		validation_data=None,
		shuffle=True,
		class_weight=None,
		sample_weight=None,
		initial_epoch=0
		)
参数：
[1]x：输入数据。若模型只有一个输入，x类型为numpy array；若模型有多个输入，x类型应为list，list的每个元素对应于各个输入的numpy array。
[2]y：标签，numpy array
[3]batch_size：整数，指定进行梯度下降时每个batch包含的样本数。训练时一个batch的样本会被计算一次梯度下降，使目标函数优化一步。
[4]epochs：整数，训练终止时的epoch值，训练将在达到该epoch值时停止，当没有设置initial_epoch时，它就是训练的总轮数，否则训练的总轮数为(epochs - initial_epoch)
[5]verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录。
[6]callbacks：list，其中的元素是keras.callbacks.Callback的对象。这个list中的回调函数将会在训练过程中的适当时机被调用，参考回调函数。
[7]validation_split：0~1之间的浮点数，用来指定训练集的一定比例数据作为验证集。validation_split用于在没有提供验证集的时候，按一定比例从训练集中取出一部分作为验证集。验证集将不参与训练，并在每个epoch结束后测试模型的指标，如损失函数、精确度等。注意，程序是先执行validation_split，再执行shuffle，故若你的数据本身是有序的，需要先手工打乱再指定validation_split，否则可能会出现验证集样本不均匀。
[8]validation_data：形式为(x, y)的tuple，是指定的验证集。此参数将覆盖validation_split。
[9]shuffle：布尔值或字符串，一般为布尔值，表示是否在训练过程中随机打乱输入样本的顺序。若为字符串"batch"，则是用来处理HDF5数据的特殊情况，它将在batch内部将数据打乱。
[10]class_weight：字典，将不同的类别映射为不同的权值，该参数用来在训练过程中调整损失函数（只能用于训练）。
[11]sample_weight：权值的numpy array，用于在训练时调整损失函数（仅用于训练）。可以传递一个1D与样本等长的向量用于对样本进行1对1的加权，或者在面对时序数据时，传递一个形式为(samples, sequence_length)的矩阵来为每个时间步上的样本赋不同的权。这种情况下请确定在编译模式时添加了sample_weight_mode="temporal"。
[12]initial_epoch：从该参数指定的epoch开始训练，在继续之前的训练时有用。
'''
```



## collections

### collections.defaultdict()

**collections.defaultdict()：**该函数返回一个类似字典的对象。该对象提供了默认值功能，defaultdict类的初始化函数接受一个类型作为参数，当所访问的键不存在的时候，可以实例化一个值作为默认值，这种形式的默认值只有在通过`dict[key]`或者`dict.getitem(key)`访问的时候才有效。
<!--在Python中如果访问字典中不存在的键，会引发KeyError异常，为避免这种情况的发生，看可以使用collections类中的defauldict()方法来为字典提供默认值，也可以通过dict.setdefault()方法来设置默认值。-->

```python
'''
collections.defaultdict([default_factory[,...]])
即collections.defaultdict(*args,**kwargs)
参数：
default_factory：即*args，提供初始值，默认为None。必须是None或可调用callable，如list、int、set。
其余参数为**kwargs，和dict构造器用法一样。
'''
from collections import defaultdict
#1、使用list作第一个参数，可以将键值对序列转换为列表字典
#当字典中没有的键第一次出现时，default_factory自动为其返回一个空列表，list.append()会将值添加进新列表；再次遇到相同的键时，list.append()将其他值再添加进该列表。
s = [('yellow',1),('blue',5),('yellow',3),('red',2),('blue',4)]
d = defaultdict(list)
for k, v in s:
    d[k].append(v)
>>>a = sorted(d.items())
[('blue', [5, 4]), ('red', [2]), ('yellow', [1, 3])]
#2、使用int作第一个参数，defaultdict可用来计数。
#函数int()，总是返回0。
s = 'mississippi'
d = defaultdict(int)
for k in s :
    d[k] += 1
>>>d
defaultdict(<class 'int'>, {'m': 1, 'i': 4, 's': 4, 'p': 2})
>>>a = sorted(d.items())
[('i', 4), ('m', 1), ('p', 2), ('s', 4)]
```

## simplejson

<!--simplejson模块主要用于将Python数据类型和json类型互相转换。-->

### simplejson.loads()

**simplejson.loads()：**解析json字符串，如json到字典转换。

```python
ret_dict = simplejson.loads(json_str)
```

### simplejson.dumps()

**simplejson.dumps()：**字典到json的转换。

```python
json_str = simplejson.dumps(ret_dict)
```

## requests

<!--requests库是一个常用于http请求的模块，可以方便的对网页进行爬取。-->
**requests：**返回一个包含服务器资源的response对象，该response对象包含返回的所有资源。

### requests.get()

**requests.get()：**通过r=requests.get(url)构造一个向服务器请求资源的url对象。该方法构造一个向服务器请求资源的requests对象。

```python
'''
requests.get(url, params=None, **kwarge)
参数：
[1]url：获取页面的url链接
[2]params：url中的额外参数，字典或字节流格式，可选
[3]**kwarge：12个控制访问的参数
'''
import requests
import simplejson

hotel_lon_lat = '116.55,39.88'
gf_key = 'sDTQHPDRYo5FaOeB9VpbJgygzq5fEMsc'
url = "http://gfapi.mlogcn.com/weather/v001/hour?lonlat={0}&hours={1}&key={2}&output_type=json".format(hotel_lon_lat, 24, gf_key)
r_json = requests.get(url)
r_dict = simplejson.loads(r_json.text)
```

### .status_code

**.status_code：**http请求的返回状态，若为200则表示请求成功。

```python
import requests
import requests
url = 'http://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word=李嘉欣&pn=0'
r = requests.get(url)
print(r.status_code)
>>
200
```

### .text

**.text：**http响应内容的字符串形式，即返回的页面内容。

```python
r.text
```

### .encoding

**.encoding：**从http header中猜测的相应内容编码方式

```python
>>>r.encoding
UTF-8
```

### .apparent_encoding

**.apparent_encoding：**从内容中分析出响应内容编码方式（备选编码方式）

```python
>>>apparent_encoding
utf-8
```

### .content

**.content：**http响应内容的二进制形式

```python
r.content
```

## time

### time.time()

**time.time()：**返回当前时间的时间戳。
【注】时间戳表示是从1970年1月1号 00:00:00开始到现在按秒计算的偏移量。查看一下`type(time.time())`的返回值类型，为float类型。

```python
import time
>>>time.time()
1665971849.2953653
```

### time.mktime()

time.mktime()：将一个struct_time转化为时间戳。
【注】time.mktime()函数执行与time.gmtime()，time.localtime()相反的操作，它接收struct_time对象作为参数，返回用秒数表示时间的浮点数。如果输入的值不是一个合法的时间，将触发OverflowError或ValueError。

```python
'''
time.mktime(t)
参数：
[1]t：结构化的时间或者完整的9位元组元素
'''
>>>time.mktime(time.localtime())
1665975387.0
```

### time.localtime()

**time.localtime()：**将一个时间戳转换为当前时区的`struct_time`，即时间数组格式的时间

```python
'''
time.localtime([secs])
参数：
[1]secs：转换为time.struct_time类型的对象的秒数。若secs参数未提供，则以当前时间为准（即会默认调用time.time()）
'''

'未给定参数'
>>>time.localtime()
time.struct_time(tm_year=2022, tm_mon=10, tm_mday=17, tm_hour=10, tm_min=1, tm_sec=49, tm_wday=0, tm_yday=290, tm_isdst=0)
'给定参数'
>>>time.localtime(1665971849.2953653)
time.struct_time(tm_year=2022, tm_mon=10, tm_mday=17, tm_hour=9, tm_min=57, tm_sec=29, tm_wday=0, tm_yday=290, tm_isdst=0)
```

### time.strftime()

**time.strftime()：**把一个代表时间的元组或者struct_time（如由time.localtime()和time.gmtime()返回）转化为格式化的时间字符串，格式由参数format决定。如果未指定，将传入time.localtime()。如果元组中任何一个元素越界，就会抛出ValueError的异常。函数返回的是一个可读表示的本地时间的字符串。

```python
'''
time.strftime(format, [, t])
参数：
[1]format：格式化字符串
[2]t：可选的参数，是一个struct_time对象。若未指定，将传入time.localtime()
'''

'通过函数将struct_time转成格式字符串'
formattime = time.localtime(1665971849.2953653)
>>>time.strftime("%Y-%m-%d %H:%M:%S", formattime)
'2022-10-17 09:57:29'

'直接使用字符串拼接成格式时间字符串'
>>>str(formattime.tm_year) + "年" + str(formattime.tm_mon) + "月" + str(formattime.tm_mday) + "日"
'2022年10月17日'

'将当前时间的时间戳转换成想要的时间格式字符串'
>>>time.strftime("%Y-%m-%d %H:%M:%S")
'2022-10-17 10:23:15'
>>>time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
'2022-10-17 10:23:34'

'获取当前时间的时分秒'
>>>time.strftime("%H:%M:%S")
'10:26:12'
'获取当前时间的年月日'
>>>time.strftime("%Y-%m-%d")
'2022-10-17'
```

​		**时间字符串支持的格式符号：（区分大小写）**

```python
%H  一天中的第几个小时（24小时制，00 - 23）       
%I  第几个小时（12小时制，0 - 11）       
%j  一年中的第几天（001 - 366）     
%m  月份（01 - 12）    
%M  分钟数（00 - 59）       
%p  本地am或者pm的相应符      
%S  秒（00 - 61）    
%U  一年中的星期数。（00 - 53星期天是一个星期的开始。）第一个星期天之    前的所有天数都放在第0周。     
%w  一个星期中的第几天（0 - 6，0是星期天）    
%W  和%U基本相同，不同的是%W以星期一为一个星期的开始。    
%x  本地相应日期字符串（如15/08/01）     
%X  本地相应时间字符串（如08:08:10）     
%y  去掉世纪的年份（00 - 99）两个数字表示的年份       
%Y  完整的年份（4个数字表示年份）
%z  与UTC时间的间隔（如果是本地时间，返回空字符串）
%Z  时区的名字（如果是本地时间，返回空字符串）       
%%  ‘%’字符  
```

### time.strptime()

time.strptime()：将格式字符串转化为struct_time。
【注】该函数是time.strftime()函数的逆操作。time.strptime()函数根据指定的格式把一个时间字符串解析为时间元组，故函数返回的是struct_time对象。

```python
'''
time.strptime(string, [, format])
参数：
[1]string：时间字符串
[2]format：格式化字符串
'''

'计算24小时以时间戳表示的时间间隔'
start_time_str = "2022-10-16 10:00:00"
end_time_str = "2022-10-17 10:00:00"

start_time_sp = time.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
end_time_sp = time.strptime(end_time_str, "%Y-%m-%d %H:%M:%S")

start_time_mk = time.mktime(start_time_sp)
end_time_mk = time.mktime(end_time_sp)

twentyfour_gap = end_time_mk - start_time_mk
>>>twentyfour_gap
86400.0
```



## Python标准库

### sys

#### exit()

python程序退出方式：os._exit()，sys.exit()

```python
'''
(1)os._exit()：直接退出，不抛出异常，不执行清理工作，余下的语句都不会执行。
(2)sys.exit(n)：抛出SystemExit异常，若没有捕获到异常，python解释器将退出；若捕获到异常的代码，那些代码会被执行。捕获这个异常可以执行清理工作。n=0为正常退出，n=其他数值(1-127)为不正常，可抛异常事件供捕获。
抛出即引发。
(3)一般情况下使用sys.exit()即可，一般在fork出来的子进程中使用os._exit()。
(4)os._exit()用于子线程中退出。
	 sys.exit()用于主线程中退出。
(5)exit(0)：无错误退出。
	 exit(1)：有错误退出。
	 退出代码是告诉解释器或操作系统的。
(6)调试代码，可以直接使用exit()，余下语句均不执行。
'''
import sys
try:
    sys.exit(0)
except SystemExit as se:
    print('处理SystemExit异常')
    print(se)
except ValueError as va:
    print('处理ValueError异常')#处理异常
    print(va)
finally:
    print('cleanup')#释放资源，无论try正常结束还是except异常结束都会执行finally代码块。
>>处理SystemExit异常
>>0
>>cleanup
```

#### sys.path

**sys.path：**指定模块的搜索路径的字符串列表。
**Tips：**sys模块包含了与python解释器和它的环境有关的函数，如sys.path属性，它是一个list，默认情况下，当python导入文件或模块时，会先在sys.path里找模块的路径，若没找到，程序会报错。

```python
'''
当要添加自己的搜索目录时，使用列表的append()方法。
对于模块和自己写的脚本不在同一个目录下，在脚本开头加sys.path.append('xxx')。
该方法是运行时修改，脚本运行后就会失效。
'''
import sys
sys.path.append('引用模块的地址')
```

### os

#### os.chdir()

**chdir()：**用于改变当前工作目录到指定的路径。

```python
'''
os.chdir(path)
参数：
path：要切换到的新路径
返回值：
允许访问返回True，否则返回False。
'''
```

#### os.getcwd()

**os.getcwd**()：获取当前执行python文件的文件夹路径。

```python
import os
>>>os.getcwd()
'E:\\Jupyterprojects'
>>>os.path.abspath("Test.ipynb")
'E:\\Jupyterprojects\\Test.ipynb'
```

#### os.curdir

**os.curdir：**与`os.getcwd()`作用相同，获取当前执行python文件的文件夹路径。

```python
import os
>>>os.curdir
'.'#表示当前路径
>>>os.path.abspath(os.curdir)
'E:\\Jupyterprojects'
```



#### os.path

##### os.path.basename()

**basename()：**返回`path`最后的文件名。若`path`以/或\结尾，那么就会返回空值。

```python
import os
pathA =  "E:\Pythonprojects\CLprediction-LSTM-simudata\data\coolingload-100.xlsx"
>>>os.path.basename(pathA)
'coolingload-100.xlsx'
```

##### os.path.abspath()

**os.path.abspath()：**返回一个目录的绝对路径。

```python
import os
>>>os.path.abspath("/Jupyterprojects")
'E:\\Jupyterprojects'
>>>os.path.abspath("/Jupyterprojects/Test.ipynb")
'E:\\Jupyterprojects\\Test.ipynb'
>>>os.getcwd()
'E:\\Jupyterprojects'
>>>os.path.abspath("Test.ipynb")
'E:\\Jupyterprojects\\Test.ipynb'
```



 ### pathlib

#### Path

##### Path.cwd()

**cwd()：**获取当前目录。

```python
from pathlib2 import Path
current_path = Path.cwd()
```

##### Path.home()

**home()：**获取Home目录。

```python
home_path = Path.home()
```

##### parent

**.parent：**获取一个路径的上级父目录。

```python
current_path.parent#获取上级父目录
current_path.parent.parent#获取上上级父目录
```

##### Path.absolute()

**absolute()：**获取绝对路径。

### json

#### json.dump()

**json.dump()：**`dump`和`dumps`对`python`对象进行序列化，将一个`python`对象进行JSON格式的编码。

```python
'''
json.dump(
    		obj,
			fp,
)
参数：
[1]obj：表示要序列化的对象。
[2]fp：文件描述符，将序列化的str保存到文件中。json模块总是生成str对象，而不是字节对象，因此，fp.write()必须支持str输入。
'''
import json
mydict={'name':'leon','age':'30','email':'xxxx@163.com'}
file='test.json'
with open(file,'w',encoding='utf-8') as f:
    json.dump(mydict,f)
    print("加载入文件完成...")
```

#### json.load()

**json.load()：**`load`和`loads`反序列化方法，将json格式数据解码为python对象。一个JSON文件经过`json.load()`以后，获得了python中的字典。

```python
'''
json.load(
			fp,
			)
参数：
[1]fp：文件描述符，将fp（.read()支持包含JSON文档的文本文件或二进制文件）反序列化为python对象。
'''
import json
filename='data\github_python_stars.json'
with open(filename,'r',encoding='utf-8') as file:
    data=json.load(file)#是file，不是filename
    print(type(data))
    #<class 'dict'>,JSON文件读入到内存以后，就是一个Python中的字典。
    # 字典是支持嵌套的，
```

### argparse

**argparse：**python自带的命令行参数解析包，用来方便地读取命令行参数。
下面是采用`argparse`从命令行获取用户名的脚本，该python的文件名为：`fun_test.py`

```python
import argparse

def main():
    parser = argparse.ArgumentParser(description="Demo of argparse")
    parser.add_argument('-n','--name', default=' Li ')
    parser.add_argument('-y','--year', default='20')
    args = parser.parse_args()
    print(args)
    name = args.name
    year = args.year
    print('Hello {}  {}'.format(name,year))

if __name__ == '__main__':
    main()
```

​		在上面的代码中，先导入了`argparse`这个包，然后包中的`ArgumentParser`类生成一个`parser`对象（常把这个叫做参数解析器），其中的`description`描述这个参数解析器是干什么的，当在命令行显示帮助信息的时候会看到`description`描述的信息。
​     接着通过对象的`add_argument()`函数来增加参数。这里我们增加了两个参数`name`和`year`，其中`'-n'`，`'--name'`表示同一个参数，`default`参数表示我们在运行命令时若没有提供参数，程序会将此值当做参数值。执行结果如下所示。

```
>python fun_test.py
Namespace(name='Li', year='20')
Hello Li 20
```

​     最后采用对象的`parse_args`获取解析的参数，由上图可以看到，`Namespace`中有两个属性（也叫成员），这里要注意个问题，当`'-'`和`'--'`同时出现的时候，系统默认后者为参数名，前者不是，但是在命令行输入的时候没有这个区分，接下来就是打印参数信息了。
当执行命令`python fun_test.py -n Wang --year '26'`结果如下：

```
>python fun_test.py -n Wang --year '26'
Namespace(n='Wang', year='26')
Hello Wang 26
```

当执行命令python fun_test.py -h可以查看帮助信息

```
>python fun_test.py -h
usage: fun_test.py [-h] [-n NAME] [-y YEAR]

Demo of argparse

optional arguments:
 -h, --help  show this help message and exit
 -n NAME, -name NAME
 -y YEAR, --year YEAR
```

<!--注：'-'为短参数，'--'为长参数-->

#### argparse.ArgumentParser()

**argparse.ArgumentParser()：**生成一个parser对象。

```python
'''
argparse.ArgumentParser(prog=None,
						usage=None,
						description=None,)
参数：
[1]prog：程序的名称（默认：sys.argv[0]）
[2]usage：描述程序用途的字符串（默认值：从添加解析器的参数生成）
[3]description：在参数帮助文档之前显示的文本（默认值：无）
'''
```



#### .add_argument()

**.add_argument()：**parser对象的方法，增加parser对象包含的参数。

```python
'''
parser.add_argument(name,
				    default,
				    type,
				    required,
				    help,)
参数：
[1]name：参数命名，如foo、-f、-foo。
[2]default：当参数未在命令行中出现时使用的值。
[3]type：命令行参数应当被转换成的类型。
[4]required：此命令行选项是否可省略（仅选项可用）
[5]help：一个此选项作用的简单描述。
'''
```



#### .parse_args()

.parse_args()：parser对象的方法，将add_argument()定义的参数进行赋值，并返回相关的namespace。

```python
>>>parser = argparse.ArgumentParser(prog='PROG')
>>>parser.add_argument('-x')
>>>parser.add_argument('--foo')
>>>parser.parse_args(['-x', 'X'])
Namespace(foo=None, x='X')

>>>parser.parse_args(['--foo', 'Foo'])
Namespace(foo='Foo', x=None)

>>>type(parser.parse_args(['--foo', 'Foo']))
argparse.Namespace

```



## Python内置函数

### next()

next()：返回迭代器的下一个项目。
<!--next()函数要和生成迭代器的iter()函数一起使用。-->

```python
'''
next(iterator[,default])
参数：
[1]iterator：迭代器
[2]default：可选，用于设置在没有下一个元素时返回该默认值；若不设置，又没有下一个元素则会触发 StopIteration 异常。
[Tips]list、tuple等都是可迭代对象iterable，通过iter()函数将iterable转换为iterator。
'''
```

### iter()

**iter()：**将可迭代对象转换为迭代器。

### enumerate()

**enumerate()：**枚举，对一个可迭代的（iterable）/可遍历的对象（如列表、字符串），enumerate将其组成一个索引序列，利用它可同时获得索引和值。
**注：**（1）enumerate多用于在for循环中得到计数。
		（2）enumerate返回的是一个enumerate对象。

```python
'''
对于一个seq，得到：
(0, seq[0]),(1, seq[1]),(2, seq[2])
'''
seq = range(5)
>>enumerate(seq)
<enumerate object at 0x000001FE7D0B24C8>

#enumerate()使用
#对一个列表，既要遍历索引又要遍历元素时
#方法1
list1 = ["这", "是", "一个", "测试"]
for i in range(len(list1)):
    print(i ,list[i])
>>>
0 这
1 是
2 一个
3 测试
#方法2
for index, item in enumerate(list1):
    print(index, item)
>>>
0 这
1 是
2 一个
3 测试

#enumerate可接收第二个参数，用于指定索引起始值
for index, item in enumerate(list1, 1):
    print(index, item)
>>>
1 这
2 是
3 一个
4 测试

#统计文件的行数
#方法1，速度慢，文件比较大时甚至不能工作
count = len(open(filepath, 'r').readlines())
#方法2
count = 0
for index, line in enumerate(open(filepath, 'r')):
    count += 1
```

### str()

**str()：**将参数转换成字符串类型。
**注：**将列表、元组、字典和集合转换为字符串后，包裹列表、元组、字典和集合的'['、']'、'('、')'、'{'、'}'，以及列表、元组、字典和集合中的元素分隔符 ','，和字典中键值对 ':' 也都转换成了字符串，是转换后字符串的一部分。

```python
#无参调用，当str()的参数省略时，函数返回空字符串。这种情况常用来创建空字符串或者初始化字符串变量。
>>str()
''
```

### zip()

zip()：将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回这些元组组成的列表。
注：如果各个迭代器的元素个数不一致，则返回列表长度与最短的对象相同。

```python
'''
zip([iterable, ...])
参数：
iterable：一个或多个迭代器
return：Python3.x，zip()返回一个zip对象，可用list()或dict()转换。Python2.x返回元组。
'''
a = [1,2,3]
b = [4,5,6,7]
a_b = zip(a, b)
>>>type(a_b)
<class 'zip'>
>>>list(a_b)
[(1, 4), (2, 5), (3, 6)]
```

### round()

**round()：**用于数字的四舍五入。

```python
'''
round(number, ndigits)
参数：
[1]digits > 0，四舍五入到指定的小数位
   digits = 0，四舍五入到最接近的整数
   digits < 0，在小数点左侧进行四舍五入
   如果round()只有number参数，等同于digits=0
注：四舍五入规则
   1、要求保留位数的后一位<=4，则进位，如round(5.214,2)保留小数点后两位，结果是 5.21
   2、要求保留位数的后一位“=5”，且该位数后面没有数字，则不进位，如round(5.215,2)，结果为5.21
   3、要求保留位数的后一位“=5”，且该位数后面有数字，则进位，如round(5.2151,2)，结果为5.22
   4、要求保留位数的后一位“>=6”，则进位。如round(5.216,2)，结果为5.22
'''
```

### float()

float()：将一个十进制整数、十进制浮点数字符串或布尔值转化为十进制浮点数。

```python
'''float(object)
参数：
[1]object：待转化成浮点数的对象
'''
>>>float(5)
5.0
>>>float(-6)
-6.0
>>>float('24.5')
24.5
>>>float('905.4')
-905.4
>>>float(True)
1.0
>>>float(False)
0.0
```



### .strip()

**.strip()：**用于去除字符串头尾指定的字符（默认为空格）或字符序列。

1. str.strip()：去除字符串两边的空格
2. str.lstrip()：去除字符串左边的空格
3. str.rstrip()：去除字符串右边的空格

注：此处的空格包含：'\n'，'\r'，'\t'，' '

```python
'''
str.strip(chars)
参数：
chars：移除字符串头尾指定的字符或字符序列，默认为空格。
'''
```

### .endswith()

**.endswith()：**判断字符串是否以指定字符或子字符串结尾。

```python
'''
str.endswith(suffix, start, end)
参数：
[1]suffix：判定后缀，为字符或字符串。为空字符时，返回值为True。
[2]start：索引字符串的起始位置，默认为0
[3]end：索引字符串的结束位置，默认为字符串str的长度len(str)
返回值：
True 或 False
'''
str = "i love python"
>>>str.endswith("n")
True
>>>str.endswith("python")
True
>>>str.endswith("n",0,6)# 索引 i love 是否以“n”结尾。
False
>>>str.endswith("")
True
>>>str[0:6].endswith("n")
False
>>>str.endswith(("z", "n"))
True
```

### any()

**any()：**当给定的可迭代参数iterable全部为False、0、空，则返回False，若有一个True，则返回True。

```python
>>>any([0])
False
>>>any([0, 1, 2])
True
>>>any([0, 1, 2, ''])
True
>>>any([0, ''])
False
>>>any((0))
Traceback (most recent call last):
  File "<input>", line 1, in <module>
TypeError: 'int' object is not iterable
>>>any((0, 1, 2))
True
```

### super()

**super()：**主要用来在子类中调用父类的方法。多用于多继承问题中，解决查找顺序（MRO）问题、重复调用问题（也称钻石继承问题或菱形图问题）等。

```python
'''
super([type[, object-or-type]])
参数：
[1]type：类，可选参数
[2]object-type：对象或类，一般是self，可选参数。

返回值：super object——代理对象
【Tips】
（1）super是一个继承自object的类，调用super()函数即是super类的实例化。
（2）super()适用于类的静态方法。
'''
#最基本的子类调用父类方法示例：
class A:
  def funxx(self):
    print("执行 A 中的 funxx 方法 ... ...")
class B(A):
  def funxx(self):
    A.funxx(self)#通过类名调用父类中的同名方法，self 参数代表 B 类的实例对象 b
    print("执行 B 中的 funxx 方法 ... ...")
>>>b = B()
>>>b.funxx()
执行 A 中的 funxx 方法 ... ...
执行 B 中的 funxx 方法 ... ...
'''
上述代码
【1】定义一个继承自A类的子类B，并在B类中重写funxx()方法，B中的funxx()是对A中funxx()功能的拓展。
【2】因为是拓展了A类的funxx()方法的功能，所以B类的funxx()方法仍然保留原功能，故可在子类B中调用父类的同名方法来实行原功能。
【3】上面的示例是通过A类类名调用A类中的同名方法来实现的，而第一个参数self实际传递的是B类的实例b。
'''

#使用super()函数来实现父类方法的调用
class A:
  def funxx(self):
    print("执行 A 中的 funxx 方法 ... ...")
class B:
  def funxx(self):
    super().funxx()
    print("执行 B 中的 funxx 方法 ... ...")
>>>b = B()
>>>b.funxx()
执行 A 中的 funxx 方法 ... ...
执行 B 中的 funxx 方法 ... ...
'''
上述代码
【1】以上执行结果和普通类名调用的结果一样。
【2】在具有单继承的类层级结构中，super()引用父类而不必显式地指定它们的名称，从而令代码更易维护。
【3】在子类中不再用父类名调用父类方法，而是用一个代理对象调用父类方法，这样当父类名改变或者继承关系发生变化时，不用对每个调用处都进行修改。
'''
```



## Python关键字

### yield

**yield：**用来暂时中止执行生成器函数并传回值。
**解析：**`yield`关键字首先是个`return`，之后再把`yield`看做是生成器（generator）的一部分（带`yield`的函数才是真正的迭代器）。
注：(1)在调用`yield`时，虽然控制权也交还给函数的调用者，但只是暂时的。
		(2)`yield`语句会暂停该函数并保留其局部状态，再次调用生成器的`next()`能够恢复执行函数。

```python
def foo():
    print('starting...')
    while True:
        res = yield 4
        print('res:',res)
g = foo()
print(next(g))
print('*'*20)
print(next(g))
>>
starting...
4
********************
res: None
starting...
4
'''
代码单步解析：
1.程序开始执行之后，因为foo函数中有yield关键字，所以foo函数并不会真的执行，而是先得到一个生成器g（相当于一个对象）。
2.直到调用next方法，foo函数正式开始执行，先执行foo函数中的print方法，然后进入while循环。
3.程序遇到yield关键字，yield相当于return，'return 4'之后，跳出foo函数，并没有执行赋值给res操作，此时next(g)语句执行完成，所以输出的前两行（第一行是while前的print，第二行是print 'return 4'的结果）是执行print(next(g))的结果。
4.之后开始执行下一个print(next(g))，执行逻辑与上一个差不多，不同的是，这个时候是从刚才那个next()停止的地方开始执行，即要执行res的赋值操作。注意，这个时候赋值操作的右边是没有值的（因为刚才那个值是return出去的，并没有给赋值操作的左边传参数），所以这个时候res赋值是None，所以输出'res:None'。
5.之后，程序继续在while里执行，又一次碰到yield，这时同样'return 4'，然后程序停止。
注：yield和return的区别：带yield的函数是一个生成器，而不是一个函数，这个生成器有一个函数即next函数，next相当于"下一步"生成哪个数，这一次的next开始的地方是接着上一次的next停止的地方执行，所以调用next的时候，生成器并不会从foo函数的第一行代码开始执行，只是接着上一步停止的地方开始，然后遇到yield后，'return'出要生成的数，此步结束。
'''
def foo():
    print('starting...')
    while True:
        res = yield 4
        print('res:',res)
g = foo()
print(next(g))
print('*'*20)
print(g.send(7))#send()发送一个参数给res
>>
starting...
4
********************
res: 7
4
'''
4.程序执行g.send(7)，程序会从上一个yield停止的地方继续向下运行，send执行赋值操作，把7赋值给res。
5.由于send()中包含next()，所以程序会继续向下运行，再次进入while循环。
6.程序执行再次遇到yield关键字，yield返回后面的值，程序再次暂停，直到再次调用next()或send()。
'''
def foo(num):
    while num<10:
        num += 1
        yield num
for n in foo(0):
    print(n)
>>
1
2
3
4
5
6
7
8
9
10
```



### return

**return：**在程序中返回某个值，返回之后程序就不再往下运行。
**注：**(1)在函数内部调用`return`语句时，控制权会永久性地交还给函数的调用者。
		(2)`return`语句会丢弃函数的局部状态。

### assert

**assert：**断言方法。

```python
def zero(s):
    a = int(s)
    assert a > 0,"a超出范围" #如果a > 0，程序正常往下运行；a < 0，抛出AssertionError：a超出范围
    return a

>>>zero("-2")
Traceback (most recent call last):
  File "<input>", line 1, in <module>
  File "<input>", line 3, in zero
AssertionError: a超出范围
```



## 其他

##### 1、os、zipfile、pathlib、time、json库为系统标准库(内部库)，无需安装

##### 2、中文乱码和坐标轴负号的处理

```python
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
```

##### 3、python 字典是无序的，无法通过数字进行索引。

##### 4、dict.items()

​		dict.items()：将字典类型转换为可遍历的元组

##### 5、python 集合（set）类型

​		python 集合（set）类型：一个**无序不重复元素集**，无法通过数字进行索引，可实现关系测试和消除重复元素功能。

##### 6、创建空dict和空set

```python
a = {}
>>> <class 'dict'>
a = {None}
>>> <class 'set'>
```

##### 7、在matplotlib中，所有可见对象都继承自Artist类，这些对象被称为Artist。

##### 8、sort()

​		sort()：用于元素的排序，默认升序，且在原列表里修改。

```python
'''
sort(key=None, reverse=False)
参数解释：
key：接收一个函数，该函数只有一个形参，函数返回一个值，值表示传入元素的权值，sort()将按照各元素的权值大小进行排序。
reverse：默认为False，表示不颠倒排列顺序；排列顺序默认为升序。
'''
list = ['aa', 'bbbb', 'ccc', 'a']
list.sort(key=f,reverse=False)
def f(a):
  return len(a)
print(list)
>>>['a', 'aa', 'ccc', 'bbbb']
```

##### 9、**%matplotlib inline**

​	（1）%matplotlib inline 是一个魔法函数。
​	（2）魔法函数定义：IPython有一组预先定义好的函数，可以通过命令行的语法形式来访问该函数。
​	（3）魔法函数分两种：一种面向行，另一种面向单元型。
​	（4）行魔法函数：用前缀 % 标注，% 后面就是魔法函数的参数，只是它的参数未写在括号或引号中来传值。
​	（5）单元型魔法函数：用前缀 %% 标注，它的参数不仅是当前 %% 行后面的内容，也包括在当前行以下的行。
​	（6）%matplotlib inline 可在 IPython 编译器如 jupyter notebook 里直接使用，功能是可以内嵌绘图，并且省略掉plt.show()。但在 spyder 或 pycharm 却无法运行，需直接注释掉。

##### 10、查看list的维度

```python
#list没有shape属性，不能直接.shape查看维度
import numpy as np
print(np.array(list).shape)
```

##### 11、python的对象维度是“从外向内”依次计算，标号从0开始。

```python
import numpy as np
np.array([1,2,3]).shape
>>>(3,)
np.array([[1],[2],[3]])
>>>(3,1)
np.array([[1,2,3]]).shape
>>>(1,3)
```

##### 12、格式描述符

```python
'''
一、宽度与精度相关描述符
(1)width.precision：整数width指定宽度，整数precision指定显示精度。
二、格式类型相关格式描述符
(1)e:科学计数格式，以e表示x10^。
'''
a = 123.456
f'{a:8.2e}'
>>1.23e+02

```

##### 13、本笔记中，">>>"表示控制台获得的结果，">>"表示脚本运行获得的结果。

##### 14、Python特色：“请求原谅比获得许可更容易(EAFP)”。

​		一个人热切地向上帝祈求自行车，即使祈求多年，上帝也没有任何回应。于是，这个人偷了一辆自行车，然后请求上帝的原谅。
​		与EAFP相对应的原则是LBYL原则（Look Before You Leap，在你跳跃前先查看）。
​		以修改字典中与键关联的值举例，LBYL原则是先使用条件语句来判断该键是否存在字典中，存在再去修改。EAFP原则是直接尝试修改与键关联的值，如果键key不存在，则抛出KeyError异常，再去根据需要进行处理。
​		相比LBYL，EAFP的优点是EAFP避免了竞跑条件错误。假设使用LBYL原则，当我们检查完键，准备修改值时，如果在其他程序删除了该字典的这个键，则会触发KeyError异常。

##### 15、程序的三种基本控制结构：顺序、循环和判断语句。

##### 16、REPL

​		REPL（“读取-求值-输出”循环，Read-Eval-Print Loop）是一个简单的、交互式的编程环境，Python的REPL就是IPython，直接在终端里定义函数，编写类，调用库，显示结果。

##### 17、Python **del**关键字

​		Python **del**关键字：用于删除对象（如类的对象、变量、列表或列表的一部分等）。

​		语法为`del object_name`。
​		由于Python是引用，del语句作用在变量上，而不是数据对象上。

```python
#del删除的是变量，而不是数据
a = 1#对象1被变量a引用，对象1的引用计数器为1
b = a#对象1被变量b引用，对象1的引用计数器加1
del a#删除变量a，解除a对1的引用
>>b#b依然引用1
1
```

##### 18、可调用的(callable)

​		可调用的(callable)：如果一个对象可以通过某种方法执行，如使用`object.(*args,**kwargs)`可执行，则这个对象是可调用的。在python中，比如函数、方法，都是可调用的。

​		可以使用系统内置函数`callable()`来判断是不是可调用的，返回True则是可调用的。

##### 19、NaN

​		NaN：Not a Number，非数字，表示未定义或不可表示的值。

注：（1）这个“不是数字”的值，是因为运算不能执行而导致，不能执行的原因要么是因为其中的运算对象之一非数字（例如，"abc"/4），要么是因为运算的结果非数字（例如，除数为0）。
		（2）虽然NaN表示“不是数字”，但它的类型是Number。
		（3）NaN和任何东西比较都是False，即使与自己比较。

##### 20、list中array的shape不一致时

​		list中array的shape不一致时：在使用numpy将list转为array的时候会报错ValueError。

```python
a = np.array([[1,2], [3,4]])
b = np.array([[5,6,7], [8,9,10]])
c = [a,b]
d = np.array(c)
>>
ValueError: could not broadcast input array from shape (2,2) into shape (2)
#broadcast广播
'''
如果一个list中有多个shape不同的array，可以先将list中的array打印出来观察。
print(c[0].shape, c[1].shape)
'''
```

##### 21、Keras的从训练集中分割出验证集

​		Keras的从训练集中分割出验证集：非测试集，使用`model.fit`。如果在`model.fit`中设置`validation_split`的值，则可将数据分为训练集和验证集，例如，设置该值为0.1，则训练集的最后10%数据将作为验证集，设置其他数字同理。注意，原数据在进行验证集分割前并没有被`shuffle`，所以这里的验证集严格就是你输入数据最末的x%。

​		如果`model.fit`的`shuffle`参数为真，训练的数据就会被随机洗乱。不设置时默认为真。训练数据会在每个epoch的训练中都重新洗乱一次。

​		验证集的数据不会被洗乱。

##### 22、状态RNN（stateful RNN）

​		状态RNN（stateful RNN）：一个RNN是状态RNN，意味着训练时每个batch的状态都会被重用于初始化下一个batch的初始状态。<!--参考Keras中文文档-->

##### 23、“冻结”网络的层

​		“冻结”网络的层：“冻结”一个层指该层的权重永不会更新。在进行fine-tune（微调）时经常会需要这项操作。<!--参考Keras中文文档-->

##### 24、Keras的模型搭建形式

​		Keras的模型搭建形式：使用符号计算方法，建立一个“计算图”。

##### 25、Keras的模型有两种

​		Keras的模型有两种：[1]Sequential，序贯模型，即单输入单输出，层与层之间只有相邻关系，跨层连接均没有，是[2]的特殊情况。[2]funtional model API，函数式模型，多输入多输出。

##### 26、深度学习的优化算法，说白了就是梯度下降。每次的参数更新有两种方式。

- 批梯度下降（Batch gradient descent）：遍历全部数据集算一次损失函数，然后算函数对各个参数的梯度，更新梯度。这种方法每更新一次参数都要把数据集里的所有样本都看一遍，计算量开销大，计算速度慢，不支持在线学习。
- 随机梯度下降（Stochastic gradient descent）：每看一个数据就算一下损失函数，然后求梯度更新参数。这种方法速度比较快，但收敛性能不太好，可能在最优点附近晃来晃去，hit不到最优点。两次参数的更新也有可能互相抵消掉，造成目标函数震荡的比较剧烈。
- 小批梯度下降（mini-batch gradient descent）：为克服以上两种方法的缺点，采用的一种折中手段。这种方法把数据分为若干个批，按批来更新参数，一个批中的一组数据共同决定本次梯度的方向，下降起来就不容易跑偏，减少了随机性。另一方面因为批的样本数与整个数据集相比小了很多，计算量也不是很大。

注：[1]基本上现在的梯度下降都是基于mini-batch，Keras的模块中的batch_size就是指这个。
		[2]Keras中用的优化器SGD（stochasticc gradient descent），但不代表是一个样本就更新一回，还是基于mini-batch。

##### 27、Keras的可训练参数在前，不可训练参数在后。

##### 28、Python路径正反斜杠（ / 和 \）的意义与用法

​		在Python中，记录路径时有以下几种写法（\n表示换行）：

```python
dir1 = r'C:\Local\Programs\Python\Python35\Lib\n_test'
dir2 = 'C:\\Local\\Programs\\Python\\Python35\\Lib\\n_test'
dir3 = 'C:/Local/Programs/Python/Python35/Lib/n_test'
```

​		这三种路径的写法时等价的：

- ​	`dir1`加入r后，使得编译器不会把'\n_test'中的"n"单独作为一个换行符，而是与后面的“_test”作为一个整体。
- ​	`dir2`中都是\\，使用双斜杠是因为其中一个反斜杠代表转义的意思，因此每一个\\都被转义成\。
- ​	`dir3`就不用自说了。

##### 29、关于空值和缺失值

- 空值：在pandas中，空值就是空字符串""
- 缺失值：`np.nan`（缺失数值），`pd.NaT`（缺失时间），或`None`（缺失字符串）。可以使用`np.nan`，`pd.NaT`，`None`创建空值。
- 还有一类数据，比如 -999、0，可能是业务上定义的缺失值。

##### 30、SettingWithCopyWarning

- SettingWithCopyWarning仅是一个警告Warning，不是错误Error，一般情况下不会影响程序的运行。
- Pandas操作的返回类型：有些操作返回原数据，有些操作返回数据的副本（Copy）
- 引起SettingWithCopyWarning警告的原因是使用了链式索引
- 链式索引是指连续使用多个索引操作，如`A[A.b==2]['c']`

```python
A = pd.DataFrame({'a':[1,2,3], 'b':[2,2,4], 'c':[2,2,5]})
A[A.b==2]['c'] = 5
>>
SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
'''
这次警告是因为A[A.b==2]['c']是链式索引，简单的说该命令直接使用了两次方括号索引。

第一次索引先执行A[A.b==2]，返回了一个DataFrame（不知道是会返回原数据A，还是返回A的副本B）；第二次索引['c']=5是找到上一步得到的DataFrame的c列对该列赋值。

但是计算机很困惑，第一步的确生成了一个DataFrame，但是第二次修改到底是修改原数据A还是A的副本B？

解决方案：使用loc或者iloc函数将两次链式操作简化为一步操作，确保第一次索引返回的是A。
'''
A.loc[A.b==2, 'c']=4
```

```python
'另一种较隐蔽的链式索引'
A = pd.DataFrame({'a':[1,2,3], 'b':[2,2,4], 'c':[2,2,5]})
B = A.loc[A.b==A.c]
>>
   a  b  c
0  1  2  2
1  2  2  2

B.loc[0, 'a'] = 2
>>
SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.

'''
这次使用了loc函数，依然出现了警告。因为虽然把链式索引分成了两个代码，但是B变量它依然可能是原始数据A的副本，也可能是原始数据A。这意味着当我们尝试修改B时，可能也修改了A。
解决方案：创建新数据B时明确告知 Pandas 创建一个副本.copy()。
'''
B = A.loc[A.b==A.c].copy()
B.loc[0, 'a'] = 2
```

##### 31、创建进入anaconda环境运行py脚本的bat文件

- 首先，右键点击Anaconda Prompt -> 更多 -> 打开文件位置 -> 右键点击Anaconda Prompt (Anaconda3) -> 属性 -> 复制目标(T) 方框内容

```python
#在编写bat文件时，只需要"/K"后面的内容
%windir%\System32\cmd.exe "/K" D:\ProgramData\Anaconda3\Scripts\activate.bat D:\ProgramData\Anaconda3
```

- 之后，创建txt文件，内容如下

```python
CALL D:\ProgramData\Anaconda3\Scripts\activate.bat D:\ProgramData\Anaconda3
CALL conda activate envname#envname为py脚本运行环境名，改为自己的
E:#切换py脚本所在盘符。盘符切换命令为“ 盘符名: ”。当bat文件所在位置与py脚本所在位置一样时，则不需要切换盘符。
cd E:\...\py_dir#进入py脚本所在路径。进入文件夹命令为 “ cd 文件夹名 ”
python py_filename.py#py_filename为要运行的py脚本名字
```

- 最后，将txt文件后缀改为bat保存，可双击运行。

##### 32.bat文件后台运行——隐藏cmd命令窗口

​		在bat文件代码头部加下面的代码即可。

```shell
@echo off
if "%1" == "h" goto begin
mshta vbscript:createobject("wscript.shell").run("""%~nx0"" h",0)(window.close)&&exit
:begin
REM
```

<!--注："REM"即python中的"#"注释符，后面跟注释内容，不运行。-->

##### 33、编译器Compiler与解释器Interpreter

- 编译器：先整体编译再执行，任何一个小改动都需要整体重新编译，可脱离编译环境运行。代表语言是C语言。
- 解释器：边解释边执行，不可脱离解释器环境运行。代表语言是Python语言。Python是动态解释性语言。

##### 34、保存的csv文件，表格中间隔有空行问题：

​			解决方法，添加`newline=''`

```python
with open(filepath, "w", newline='') as f_obj:
```

##### 35、Python中main方法中的变量使用

​		`__main__`方法作为Python编程环境的程序入口，可以定义一些变量，这些变量默认为全局变量，可供其他地方使用。

```python
#windows命令窗口执行脚本
import argparse

def test_fun1():
    print("pathA:", pathA)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Option for test.")
    parser.add_argument("--pathA")
    args = parser.parse_args(["--pathA","hello"])
    pathA = args.pathA
    
    test_fun1()
```

​		在上述代码中，`pathA`在`__main__`方法中定义，默认为全局（global）变量，可以供`test_fun1()`方法调用，可以正常打印。

```python
#错误修改__main__方法中的变量
import argparse
import os

def test_fun2():
    if pathA.endswith('/'):
        pathA = os.path.basename(pathA[:-1])#尝试修改pathA，报错
    else:
        pathA = os.path.basename(pathA)
    print("pathA:", pathA)
if __name__ == "__main__":
    parser = argparse.ArgumentParser('Options for test.')
    parser.add_argument("--pathA")
    args = parser.parse_args(["--pathA", "E:\Pythonprojects\CLprediction-BLS-demo"])
    pathA = args.pathA
    
    test_fun2()

 >>
UnboundLocalError: local variable 'pathA' referenced before assignment
```

​		在上述代码中，`pathA`在`__main__`方法中定义，默认为全局（global）变量，如果其他方法调用`pathA`时，未加`global`声明，所调用的`pathA`视为常量。
​		在`test_fun2()`方法中，对`pathA`进行修改后赋值，此时`pathA`会被解释器解释为`test_fun2()`方法的局部变量，因此会报错。
​		使用下述方法进行全局变量修改。

```python
#正确修改__main__方法中的变量
import argparse
import os

def test_fun3():
    if pathA.endswith('/'):
        pathA_name = os.path.basename(pathA[:-1])#pathA切片，不会改变pathA变量本身
    else:
        pathA_name = os.path.basename(pathA)
    print("pathA_name:", pathA_name)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser('Options for test.')
    parser.add_argument("--pathA")
    args = parser.parse_args(["--pathA", "E:\Pythonprojects\CLprediction-BLS-demo/"])
    pathA = args.pathA
    
    test_fun2()
>>
pathA_name: CLprediction-BLS-demo
```

​		使用`global`进行全局变量声明。除了在`__main__`中定义全局变量外，还可以在文件开头使用`global`显式声明全局变量。通常不这么做，一般仅会在文件头使用全大写变量名定义全局常量。

```python
def test1():
    print(pathA)
    pathA = "xxx"
    print(pathA)
    
def test2():
    print(pathA)
    global pathA
    pathA = "xxx"
    print(pathA)
    
def test3():
    global pathA
    print(pathA)
    pathA = "xxx"
    print(pathA)
    
if __name__ == "__main__":
    pathA ="aaa"
    
    #test1()运行结果
    test1()
    >>
    UnboundLocalError: local variable 'pathA' referenced before assignment
    
    #test2()运行结果
    test2()
    >>
    SyntaxError: name 'pathA' is used prior to global declaration
    
    #test3()运行结果
    test3()
    >>
    aaa
    xxx
```

##### 36、python遇到库版本不对

​		与其更新库，不如卸载库，装过所需版本的库，更新库会残留以前的版本，导致版本交叉干扰，直接找出相应文件夹删除。

##### 37、超参数寻优方法选择

​		除非要研究的超参数值很少，否则应优先选择随机搜索而不是网格搜索。如果训练时间很长，则选择贝叶斯优化方法。

##### 38、使用回调函数

​		【ModelCheckpoint回调】
​		如果在训练期间使用验证集，则可以在创建ModelCheckpoint时设置save_best_only=True。
​		在这种情况下，只有在验证集上的模型性能达到目前最好时，它才会保存模型。这样就不必担心训练时间过长而过拟合训练集：只需还原训练后保存的最后一个模型，这就是验证集中的最佳模型。
​		【EarlyStopping回调】
​		EarlyStopping回调会在多个轮次（即patience次）的验证集上没有任何进展时，中断训练，并且可以选择回滚到最佳模型。
​		【结合ModelCheckpoint和EarlyStopping】
​		可将两个回调结合起来以保存模型的检查点，并在没有更多进展时尽早中断训练。
​		可以将轮次设置为较大的值，因为训练将在没有更多进展时自动停止。在这种情况下，无需还原保存的最佳模型，因为EarlyStopping回调将跟踪最佳权重，并在训练结束时还原它。

##### 39、axis=0&axis=1

​		axis = 0 代表对横轴（行）操作，也就是第0轴
​		axis = 1 代表对纵轴（列）操作，也就是第1轴
​		**操作为：**
​		axis= 0 对a的横轴（行）进行操作，在运算的过程中其运算的方向表现为纵向（列）运算
​		axis= 1 对a的纵轴（列）进行操作，在运算的过程中其运算的方向表现为横向（行）运算

##### 40、Early stopping 的 val_loss不可见

​		问题描述：
​	tensorflow:Early stopping conditioned on metric val_loss which is not available. Available metrics are: loss,accuracy
​		解决方法：
​	增加验证集大小，使可看见训练过程中loss、accuracy、val_loss和val_accuracy

##### 41、查看函数帮助信息

```python
import pandas as pd
help(pd.to_datetime)#不是pd.to_datetime()，没有括号
```

##### 42、np.array数据格式做减法

```python
#两数维度不同时，相减
a = np.array([[1],[2]])
b = np.array([3, 4, 5])
>>>a.shape
(2, 1)
>>>b.shape
(3,)
>>>a-b
[[-2 -3 -4]
 [-1 -2 -3]]
>>>b-a
[[2 3 4]
 [1 2 3]]
>>>(a-b).shape
(2, 3)
```

##### 43、在keras+Tensorflow中使用自定义函数时，出现ValueError: Unknown metric function: ***

​		在使用Tensorflow+Keras自定义损失函数进行训练时，出现ValueError: Unknown metric function，这是因为自定义的函数没有被保存，解决方法如下：

```python
def last_time_step_mse(Y_true, Y_pred):
    return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])
#部分代码省略
model.compile(
optimizer="adam",
loss="mse",
metrics=[last_time_step_mse])
#部分代码省略
model.save('CLprediction.h5')
'如果此时存储的model中有自定义的loss或者metrics时'
#单单这样加载模型会报错ValueError，因为自定义的函数没有被保存
model = load_model("CLprediction.h5")
'应该使用如下加载方式'
model = load_model("CLprediction.h5", 
                   custom_objects={'last_time_step_mse':last_time_step_mse})
```

##### 44、DataFrame重建索引

​		设置`ignore_index`：默认为False，为True时重建索引。

##### 45、tensorflow自带keras

​		无需安装keras。

##### 44、报错：“cannot import name ‘dtensor‘ from ‘tensorflow.compat.v2.experimental’”

​		在使用`pip install tensorflow-cpu==2.6.0`时，会自动附带安装`keras=2.10.`，此时运行代码时，会因为keras版本太高，而报错，需要降低到和tensorflow版本一致，使用`pip install keras==2.6`，降低keras版本。

##### 45、将pip源临时更换为源

```python
'更换为清华源，package为包名'
pip install package -i https://pypi.tuna.tsinghua.edu.cn/simple
'更换为中科大源'
pip install package -i https://mirrors.ustc.edu.cn/pypi/web/simple/
'更换为豆瓣源'
pip install package -i http://pypi.doubanio.com/simple/
'更换为阿里云源'
pip install package -i http://mirrors.aliyun.com/pypi/simple/
'下述为例子'
pip install tensorflow-cpu==2.6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

##### 46、Matplotlib升级到3.6后程序执行告警MatplotlibDeprecationWarning的解决方法

```python
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('qt5agg')
```

##### 47、tensorflow2_clts环境安装

```c
conda create --name tensorflow2_clts python=3.8
conda activate tensorflow2_clts
pip install tensorflow
conda install pandas
conda install matplotlib
conda install -c conda-forge bayesian-optimization
conda install openpyxl
```

```python
#该环境需要在代码使用：
import matplotli
matplotlib.use(‘qt5agg’)
```



# shape-based-matching代码梳理



​	项目文件编译后，达到可执行状态，如下图所示，算法支持模板的旋转匹配，这里作者是将方向分为360,将各方向的模板都进行保存与匹配。


```C++
test img size: 1044480
construct response map
elasped time:0.0893522s
templ match
elasped time:0.00432388s
elasped time:0.0960523s
matches.size(): 7

match.template_id: 340
match.similarity: 98.6641
test end
```

下面将从angle_test的执行逻辑进行分析源代码。

## 1. detector.readClasses

readClasses使用cv::FileStorage加载yaml文件，生成文件根节点对象fs.root()，传入Detector::readClass函数。这个阶段主要做了文件读取的准确。yaml文件主要读取了info和template，template文件每一个模板对象使用一个map，拥有template_id和templates列表，这个列表包含了不同pyramid_level模板信息，每一个模板的features元素是包含了坐标和对应的梯度编码，这里作者没有使用5个方向的优化方案，选择了8个方向。


readClass将yaml数据按照上面的关键字信息与组织结构存入`std::vector<TemplatePyramid> &tps` ，这里vector整体作为class_id的value。



## 2. shape_based_matching::shapeInfo_producer::load_infos

加载test_iofo.yaml，生成 `std::vector<Info> infos` 其中：

```C++
class Info{
public:
    float angle;
    float scale;

    Info(float angle_, float scale_){
        angle = angle_;
        scale = scale_;
    }
};
```



## 3. 读取图片预处理

1. 使用opencv读取搜索图片；
2. 指定padding大小，生成padding图像；
3. 指定步长，裁剪出能被步长整除的padding图像；
4. 初始化定义 `line2Dup::Detector detector(128, {4, 8});` detector是我们进行匹配的算子对象；

## 4. quantizedOrientations

1. 平滑图像

   1. ```C++
      Mat smoothed;
      // Compute horizontal and vertical image derivatives on all color channels separately
      static const int KERNEL_SIZE = 7;
      // For some reason cvSmooth/cv::GaussianBlur, cvSobel/cv::Sobel have different defaults for border handling...
      GaussianBlur(src, smoothed, Size(KERNEL_SIZE, KERNEL_SIZE), 0, 0, BORDER_REPLICATE);
      ```

   2.   这里定义了高斯模糊卷积和大小为7

2. 后面分彩色图片和灰度图片进行处理，这里只剖析灰度图的执行逻辑，彩色图类似；

3. `Mat sobel_dx, sobel_dy, sobel_ag;` 分别定义了X方向梯度图，Y方向梯度图，以及梯度方向图；

4. `Sobel(smoothed, sobel_dx, CV_32F, 1, 0, 3, 1.0, 0.0, BORDER_REPLICATE);`求X方向梯度图；Y方向类似；

5. `magnitude = sobel_dx.mul(sobel_dx) + sobel_dy.mul(sobel_dy);`求梯度幅值图；

6. `phase(sobel_dx, sobel_dy, sobel_ag, true);` 由cv::phase求梯度的角度图；

7. `hysteresisGradient(magnitude, angle, sobel_ag, threshold * threshold);` 离散量化梯度方向；

8. 同步修改了detector.match函数中定义的`quantizers`

## 5. hysteresisGradient

1. 通过cv::convertTo放缩方向角度，量化为16个方向(实际是8个方向)；

   1. ```C++
      Mat_<unsigned char> quantized_unfiltered;
      angle.convertTo(quantized_unfiltered, CV_8U, 16.0 / 360.0);
      ```

2. 第一行，最后一行的梯度值置为0

   1. ```C++
      memset(quantized_unfiltered.ptr(), 0, quantized_unfiltered.cols);
      memset(quantized_unfiltered.ptr(quantized_unfiltered.rows - 1), 0, quantized_unfiltered.cols);
      ```

3. 每一行第一列和最后一列值置为0

   1. ```C++
      for (int r = 0; r < quantized_unfiltered.rows; ++r)
      {
          quantized_unfiltered(r, 0) = 0;
          quantized_unfiltered(r, quantized_unfiltered.cols - 1) = 0;
      }
      ```

4. 将16个方向映射为8个方向

   1. ```C++
      for (int r = 1; r < angle.rows - 1; ++r)
      {
          uchar *quant_r = quantized_unfiltered.ptr<uchar>(r);
          for (int c = 1; c < angle.cols - 1; ++c)
          {
              quant_r[c] &= 7;  // 按位与操作
          }
      }
      ```

5. 筛选梯度赋值比较显著的点，低于阈值的点过滤

6. `quantized_angle = Mat::zeros(angle.size(), CV_8U);`quantized_angle是入参，也是潜在的出参

7. 幅值大于阈值时，修改quantized_angle

   1. 初始化一个直方图统计数组：`int histogram[8] = {0, 0, 0, 0, 0, 0, 0, 0};`

   2. 统计当前像素九宫格内的方向分布：

      1. ```C++
         // 九宫格上一行第一个元素指针
         uchar *patch3x3_row = &quantized_unfiltered(r - 1, c - 1);
         histogram[patch3x3_row[0]]++;
         histogram[patch3x3_row[1]]++;
         histogram[patch3x3_row[2]]++;
         // 九宫格当前行第一个元素指针
         patch3x3_row += quantized_unfiltered.step1();
         histogram[patch3x3_row[0]]++;
         histogram[patch3x3_row[1]]++;
         histogram[patch3x3_row[2]]++;
         // 九宫格下一行第一个元素指针
         patch3x3_row += quantized_unfiltered.step1();
         histogram[patch3x3_row[0]]++;
         histogram[patch3x3_row[1]]++;
         histogram[patch3x3_row[2]]++;
         ```

8. 九宫格大部分方向一致的才给出梯度方向

   1. ```C++
      static const int NEIGHBOR_THRESHOLD = 5;
      if (max_votes >= NEIGHBOR_THRESHOLD)
          quantized_angle.at<uchar>(r, c) = uchar(1 << index);
      ```

   2.   默认是九宫格有5个相同的梯度方向才给最终的梯度方向。

## 6. orUnaligned8u

orUnaligned8u使用了MIPP库，这个项目很多地方都使用了这个库，MIPP即我的内部库，但是这个库不是作者写的，而是使用的开源库：https://github.com/aff3ct/MIPP/tree/master/include

1. `mipp::N<uint8_t>()`指定了可并发处理uint8_t数据类型的数量；(Github上有明确说明)
2. mipp::Reg<uint8_t> src_v((uint8_t*)src + c); Reg是向寄存器注册向量src_v，数据量由mipp::N<uint8_t>控制；
3. dst_v同理；
4. mipp::orb是在寄存器上对数据进行按位或操作；
5. res_v.store((uint8_t*)dst + c);中mipp::Reg::store是存数据到dst指向的Mat

这里实现了论文中合并周围梯度方向的功能：

按位或之后相当于得到了binarized image.

## 7. computeResponseMaps

reponse_maps是一个Mat向量，函数内部定义size大小为8, cv::Mat的size是binarized image的size。

lsb4标记binarized image元素的后四位字节内容；

msb4标记binarized image元素的前四位字节内容；(240的二进制表示为“0b11110000”)

低四位和高四位分别制表：

根据binarized image得到每个点的梯度得分。

## 8. 使用detector.match生成Match对象向量

1. `Timer timer;`是初始化一个计时对象，match结束后会输出match的时间。
2. `std::vector<Match> matches;`定义返回体。
3. `std::vector<Ptr<ColorGradientPyramid>> quantizers;`定义金字塔梯度对象向量，；这里项目默认size是1；
4. `cv::Ptr<ColorGradient> modality;` `Detector`内定义了
5. `quantizers.push_back(modality->process(source, mask));`添加一个`Ptr<ColorGradientPyramid>`对象进`quantizers`,同时注意内部构造函数已经实现了quantizedOrientations的调用，即已经对搜索图求了梯度。这里使用高斯模糊，再在x方向、y方向求sobel梯度，使用phase转笛卡尔坐标系到极坐标系，得到角度与幅值。最后调用hysteresisGradient函数，量化所有梯度方向得到quantized_angle，quantized_angle即quantizers.angle。同时记录了quantizers.magnitude(梯度幅值)，quantizers.angle_ori(sobel梯度方向)
6. `LinearMemoryPyramid `**`lm_pyramid`**`(pyramid_levels, std::vector<LinearMemories>(1, LinearMemories(8)));`其中LinearMemoryPyramid是嵌套向量容器类型(`Detector`自定义)；pyramid_levels是 `3.d` 步骤指定的私有变量，根据上面的指定，pyramid_levels等于2。内部向量容器大小为1，类型为 `cv::Mat`，这里LinearMemories指定大小为8（8对应着上面绘制的朴素梯度方向）。
7. `for (int l = 0; l < pyramid_levels; ++l) `开始对图像金字塔进行循环计算
8. `int T = T_at_level[l];`确定当前层对应的缩放倍数
9. `std::vector<LinearMemories> &lm_level = lm_pyramid[l];`指定当前层的梯度cv::Mat的指针
10. TODO：if条件语句
11. `Mat quantized, spread_quantized; `定义量化的梯度Mat和离散化的梯度Mat
12. `std::vector<Mat> response_maps; `定义返回的查找得分Mat
13. `quantizers[i]->quantize(quantized); ` `quantized`量化初始化，初始化一个全0的图，将quantizers[0].angle根据quantizers[0].mask拷贝到quantized。前面quantizers没有使用[0]取值，只是为了理解简写了。此时quantized是mask区域的量化梯度，量化到8个方向，过滤九宫格范围内没有5个一致梯度方向的像素点。
14. `spread(quantized, spread_quantized, T);`再次离散梯度，操作spread_quantized，内部主要调orUnaligned8u实现，spread_quantized即Binarized Image。
15. computeResponseMaps计算得分图

## 9. linear memories

我们上面设置的T是{4, 8}，如图所示：左边将搜索图的所有梯度方向得分都分块进行线性拉伸存储，即得到T方个线性存储单元(对于某个梯度方向)，对于模板template，同样分块，根据梯度方向到对应的线性存储单元集中进行索引，根据块和梯度在块中的坐标获取线性单元集的某个线性存储单位和偏置。这样可以得到搜索图的得分Mat, 求最大值即可获取对应的最佳匹配位置和相似度评分。



致谢：文中部分图片采用网络资源, 感谢大佬们的开源资源帮助！

1. https://blog.csdn.net/weixin_41864918/article/details/124324107

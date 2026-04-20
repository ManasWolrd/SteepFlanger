# iir
使用切比雪夫1型滤波器应该会有点意思，用并行滤波器结构看看能不能降低分数延迟线带来的click，以及更容易进行SIMD优化

## chebyshev type I 离散极点
令N=多少个biquad滤波器，阶数=2N  
通带涟漪dB=ripple  

模拟的极点计算，注意pole和zero都是共轭形式，这里只写了一个im符号对应的极点

$eps = sqrt(10^(ripple/10) - 1)$  
$A = 1/(2N) * asinh(1/eps)$  
$k_re = sinh(A)$  
$k_im = cosh(A)$  

$i = [0, N)$  
$phi[i] = \frac{(2i+1)\pi}{4N}$  
```
pole[i] = {  
    $real: k_re * -sin(phi[i])$  
    $imag: k_im * cos(phi[i])$  
}
```

OMG，差点忘记了调整低通滤波器的频率  
$wc = [0, pi]$
$pole[i] = wc * pole[i]$

使用双线性变换  
令采样率$fs = 0.5$  
$k = 2fs = 1$  
$zpole[i] = (k + pole[i]) / (k - pole[i])$  
$zzero[i] = -1$  
$$zk[i] = \frac{wc*wc}{\Re{(1 - pole[i]) * (1 - conj[pole[i]])}}$$

## residualz 转并行
现在的滤波器形式是
$$H(z) = \prod_{i=0}^{N-1} \frac{(z+1)^2}{(z-zpole)(z-conj\{zpole\})}
       = \prod_{i=0}^{N-1} \frac{(z^-1+1)^2}{(1-zpole*z^-1)(1-conj\{zpole\}*z^-1)} $$

计算第i个极点对应的留数为H(zpole)(z-zpole)，在实现的时候分母运算时不计算，只计算其他的极点  

现在应该得到了
$$\sum_{i=0}^{N-1} \frac{r}{z-p} \frac{conj[r]}{z-conj[p]} = \frac{2Re[r]z-2Re[r*conj[p]]}{z^2-2Re[p]z+norm[p]}$$

> 我还以为常数项可以这样算，你要是用scipy运算的话真的是可以得到下面这个等式的，但我用的遮盖法如果这样计算反而是错误的  
> 因为len(zpole) = len(zzero)  
> 常数项$C = \lim_{z^-1 \to 0} H(z^-1) - \sum_{i=0}^{N-1} 2\Re{\{r[i]\}} = 1-\sum_{i=0}^{N-1} 2\Re{\{r[i]\}} $

因为len(zpole) = len(zzero)  
常数项$C = \lim_{z^-1 \to 0} H(z^-1) = 1 $

## 使用 DF系列
$b0 = 0$   
$b1 = 2Re[r]$  
$b2 = -2Re[r*conj[p]]$  
$a0 = 1$  
$a1 = -2Re[p]$  
$a2 = norm[p]$  

## 使用分数定义式
$$\frac{rz^-1}{1-pz^-1}$$
这里只有半个极点，要得到实数结果，只需要加上滤波器输出的Real部分*2即可

## 使用 ZDF-SVF
TODO

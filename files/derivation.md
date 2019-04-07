To understand the derivation of the Hebbian, anti-Hebbian network written in the paper, I derived it again by myself.



The cost function we would like to optimize is:
$$
y_T = \underset{y_{T} \ge 0}{\arg \min } \left\| X'X - Y'Y \right\|_F^2 ＋\lambda \, \mathrm{rank}(Y)
$$

$$
\begin{align}
y_T &= \underset{y_T \ge 0, Card(y_T) \ge Card(y_{T-1})}{\arg \min }
\left[   \sum_{t=1}^T \sum_{s=1}^T (x'_t x_s - y'_t y_s)^2  + \lambda \, Card(y_T) \right] \\
&= \underset{y_T \ge 0, Card(y_T) \ge Card(y_{T-1})}{\arg \min } \left[ 2  \sum_{t=1}^{T-1}(x'_t x_T - y'_t y_T)^2 + (x'_T x_T - y'_T y_T)^2 + \lambda \, Card(y_T) \right] \\
&= \underset{y_T \ge 0, Card(y_T) \ge Card(y_{T-1})}{\arg \min }
\left[
 2  \sum_{t=1}^{T-1}((x'_t x_T)^2 - 2x'_t x_Ty'_t y_T +(y'_t y_T)^2 ) + \\
 ((x'_T x_T)^2 - 2x'_T x_Ty'_T y_T + (y'_T y_T)^2) + \\
 + \lambda \, Card(y_T)
\right] \\
\end{align}
$$

We can remove terms which are not related with  $y_T$, so


$$
\begin{align}
&= \underset{y_T \ge 0, Card(y_T) \ge Card(y_{T-1})}{\arg \min }
\left[
 2  \sum_{t=1}^{T-1}(-2x'_t x_Ty'_t y_T +(y'_t y_T)^2 ) \\
 -2x'_T x_Ty'_T y_T + (y'_T y_T)^2 + \\
 + \lambda \, Card(y_T)
\right] \\
&= \underset{y_T \ge 0, Card(y_T) \ge Card(y_{T-1})}{\arg \min }
\left[
  \sum_{t=1}^{T-1}(-4x'_t x_Ty'_t y_T +2(y'_t y_T)^2 ) \\
 -2x'_T x_Ty'_T y_T + (y'_T y_T)^2 + \\
 + \lambda \, Card(y_T)
\right] \\
\end{align}
$$

$x$ is $n$-dim vector and $y$ is $m$-dim vector.

$$
\begin{align}
&= \underset{y_T \ge 0, Card(y_T) \ge Card(y_{T-1})}{\arg \min }
\left[
  \sum_{t=1}^{T-1} \{ (-4 \sum_{j=1}^n{x_{tj}}{x_{Tj}}) (\sum_{i=1}^m{y_{ti}}{y_{Ti}}) +2(\sum_{i=1}^m{y_{ti}}{y_{Ti}})^2  \} \\
 -2(\sum_{j=1}^n{x_{Tj}}{x_{Tj}}) (\sum_{i=1}^m{y_{Ti}}{y_{Ti}}) + (\sum_{i=1}^m{y_{Ti}}{y_{Ti}} )^2 + \\
 + \lambda \, Card(y_T)
\right] \\
\end{align}
$$


$$
\begin{align}
&= \underset{y_T \ge 0, Card(y_T) \ge Card(y_{T-1})}{\arg \min }
\left[
  -4 \sum_{i=1}^m {y_{Ti}} \sum_{j=1}^n {x_{Tj}}\sum_{t=1}^{T-1}{y_{ti}}{x_{tj}}　\\ 
  +2 \sum_{i=1}^m y_{Ti} \sum_{k=1}^m y_{Tk} \sum_{t=1}^{T-1}  {y_{ti}} {y_{tk}} \\
  -2 \sum_{i=1}^m \parallel x_T \parallel^2_2 {y_{Ti}}^2 + \\
     \sum_{i=1}^m {y_{Ti}}^2 \sum_{k=1}^m {y_{Tk}^2} + \\
 + \lambda \, Card(y_T)
\right] \\
&= \underset{y_T \ge 0, Card(y_T) \ge Card(y_{T-1})}{\arg \min }
\sum_{i=1}^m \left[
  -4 {y_{Ti}} \sum_{j=1}^n {x_{Tj}}\sum_{t=1}^{T-1}{y_{ti}}{x_{tj}}　\\ 
  +2  y_{Ti} \sum_{k=1}^m y_{Tk} \sum_{t=1}^{T-1}  {y_{ti}} {y_{tk}} \\
 -2  \parallel x_T \parallel^2_2 {y_{Ti}}^2 + \\
 {y_{Ti}}^2 \sum_{k=1}^m {y_{Tk}^2} \\
\right] + \lambda \, Card(y_T)
\end{align}
$$

We think about optimizing this by applying coordinate descent for each dimension of $y_{T}$.

$$
\begin{align}
y_{Ti} &= \underset{y_{Ti} \ge 0, Card(y_T) \ge Card(y_{T-1})}{\arg \min }
\left[
  -4 {y_{Ti}} \sum_{j=1}^n {x_{Tj}}\sum_{t=1}^{T-1}{y_{ti}}{x_{tj}}　\\ 
  +4  y_{Ti} \sum_{k=1, k\ne i}^m y_{Tk} \sum_{t=1}^{T-1} {y_{ti}} {y_{tk}} + 
2 y_{Ti}^2 \sum_{t=1}^{T-1}  {y_{ti}^2} \\
-2 \parallel x_T \parallel^2_2 {y_{Ti}}^2 + \\
 2{y_{Ti}}^2 \sum_{k=1,k\ne i}^m {y_{Tk}^2 + {y_{Ti}}^4} 
 \right] \\
 &= \underset{y_{Ti} \ge 0, Card(y_T) \ge Card(y_{T-1})}{\arg \min }
\left[
  -4 {y_{Ti}} \sum_{j=1}^n {x_{Tj}}\sum_{t=1}^{T-1}{y_{ti}}{x_{tj}}　\\ 
  +4  y_{Ti} \sum_{k=1, k\ne i}^m y_{Tk} \sum_{t=1}^{T-1} {y_{ti}} {y_{tk}} + 
2 y_{Ti}^2 \sum_{t=1}^{T-1}  {y_{ti}^2} \\
\Biggl( 2\sum_{k=1,k\ne i}^m {y_{Tk}^2} -2 \parallel x_T \parallel^2_2 \Biggr) {y_{Ti}}^2 + {y_{Ti}}^4
\right]
\end{align}
$$

If the $i$-th node of output $y_T$ has not been utilized during $t=1$ to  $T-1$, i.e. $\sum_{t=1}^{T-1} y_{ti}^2 = 0$, we adjust the output as

$$
y_{Ti}=\underset{y_{Ti} \ge 0}{\arg \min }\left[\left(\left\|x_{T}\right\|^{2}_2-\sum_{k=1, k \neq i}^{m} y_{Tk}^{2}\right)-y_{Ti}^{2}\right]^{2}+\lambda\left\|y_{Ti}\right\|_{0}=\\
\left\{
\begin{array}{l}
0, \quad\left(\left\|x_{T}\right\|^{2}_2-\sum_{k=1, k \neq i}^{m} y_{Tk}^{2}\right)^{2} \leq \lambda \\
\left(\left\|x_{T}\right\|^{2}_2-\sum_{k=1, k \neq i}^{m} y_{Tk}^{2}\right)^{1 / 2}, \left(\left\|x_{T}\right\|^{2}_2-\sum_{k=1, k \neq i}^{m} y_{Tk}^{2}\right)^{2}>\lambda
\end{array}
\right.
$$

Once the $i​$-th output node becomes active, then $\sum_{t=1}^{T-1} y_{ti}^2 > 0​$, so can write the equation as

$$
\begin{align}
y_{Ti} &= 
\underset{y_{Ti} \ge 0}{\arg \min }
\sum_{t=1}^{T-1} {y_{ti}^2}  \left[ 
- \frac{ 4 {y_{Ti}} \sum_{j=1}^n {x_{Tj}}\sum_{t=1}^{T-1}{y_{ti}}{x_{tj}}}{{\sum_{t=1}^{T-1} y_{ti}^2}}　\\ 
+ \frac {4 y_{Ti} \sum_{k=1, k\ne i}^m y_{Tk} \sum_{t=1}^{T-1} {y_{ti}} {y_{tk}}} 
  { {\sum_{t=1}^{T-1} y_{ti}^2} } + 
2 y_{Ti}^2 \\
 \frac{ 2 \Biggl( \sum_{k=1,k\ne i}^m {y_{Tk}^2} - \parallel x_T \parallel^2_2 \Biggr) {y_{Ti}}^2}{{\sum_{t=1}^{T-1} y_{ti}^2}}
+ \frac{{y_{Ti}}^4}{{\sum_{t=1}^{T-1} y_{ti}^2}} 
\right] \\
&= \underset{y_{Ti} \ge 0}{\arg \min }
\sum_{t=1}^{T-1} {y_{ti}^2}  
\left[ 
- 4 {y_{Ti}} \sum_{j=1}^n \Biggl(\frac{  \sum_{t=1}^{T-1}{y_{ti}}{x_{tj}}}{{\sum_{t=1}^{T-1} y_{ti}^2}}\Biggr) {x_{Tj}}\\ 
+ 4 y_{Ti} \sum_{k=1, k\ne i}^m \Biggl(\frac { \sum_{t=1}^{T-1} {y_{ti}} {y_{tk}}} 
{ {\sum_{t=1}^{T-1} y_{ti}^2} } \Biggr) y_{Tk}+ 
2 y_{Ti}^2 \\
 \frac{ 2 \Biggl( \sum_{k=1,k\ne i}^m {y_{Tk}^2} - \parallel x_T \parallel^2_2 \Biggr) {y_{Ti}}^2}{{\sum_{t=1}^{T-1} y_{ti}^2}}
+ \frac{{y_{Ti}}^4}{{\sum_{t=1}^{T-1} y_{ti}^2}} 
\right]
\end{align}
$$

At the large $T$ limit, $\sum_{t=1}^{T-1} y_{ti}^2​$ in the denominators of the 3rd and 4th term become large.  So we can ignore the 3rd and 4th term.

$$
\begin{align}
y_{Ti} &\approx \underset{y_{Ti} \ge 0}{\arg \min }
\sum_{t=1}^{T-1} {y_{ti}^2}  
\left[ 
- 4 {y_{Ti}} \sum_{j=1}^n \Biggl(\frac{  \sum_{t=1}^{T-1}{y_{ti}}{x_{tj}}}{{\sum_{t=1}^{T-1} y_{ti}^2}}\Biggr) {x_{Tj}}\\ 
+ 4 y_{Ti} \sum_{k=1, k\ne i}^m \Biggl(\frac { \sum_{t=1}^{T-1} {y_{ti}} {y_{tk}}} 
{ {\sum_{t=1}^{T-1} y_{ti}^2} } \Biggr) y_{Tk}+ 
2 y_{Ti}^2 
\right] \\
&\approx \underset{y_{Ti} \ge 0}{\arg \min }
\sum_{t=1}^{T-1} {y_{ti}^2}
\left[ 
- 4 {y_{Ti}} \sum_{j=1}^n W_{Tij} {x_{Tj}}\\ 
+ 4 y_{Ti} \sum_{k=1, k\ne i}^m M_{Tik} y_{Tk}+ 
2 y_{Ti}^2 
\right] 
\end{align}
$$

Here we used the substituion of $W,M$ as 

$$
W_{Tij} = \frac{  \sum_{t=1}^{T-1}{y_{ti}}{x_{tj}}}{{\sum_{t=1}^{T-1} y_{ti}^2} } \\
M_{Tik\ne i} = \frac { \sum_{t=1}^{T-1} {y_{ti}} {y_{tk}}} { {\sum_{t=1}^{T-1} y_{ti}^2} }; M_{Tii}=0
$$

This optimization can be treated as

$$
y_{Ti} \approx \underset{y_{Ti} \ge 0}{\arg \min } (W_{Ti} {x_{Tj}} - M_{Ti} y_{T} - y_{Ti})^2
$$

And we obtain,

$$
y_{Ti} = \max(W_{Ti} {x_{T}} - M_{Ti} y_{T}, 0)
$$



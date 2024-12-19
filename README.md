# README
进入wsl的命令行：
```shell
cd '/mnt/e/OneDrive - sjtu.edu.cn/6-SoftwareFiles/GitFiles/0-个人库/03-科研'
tree . -L 3
```

```shell
```
## 修改记录
- 2024-12-11: 构建CS2的数据集
- 2024-12-12: 构建域、粒度和解释方法的参数框架
- 2024-12-13: 编写Exchange、ExchangeV2、ExchangeV3的代码并测试效果
- 2024-12-14: 发现ExchangeV2下不同样本的attribution趋于一致，由于self.model未使用eval模式
- 2024-12-15: 发现同类交换（a类patch引入a类样本）也会导致显著预测下降，不符合预期；经相位分离后成功解决。
- 2024-12-16: 发现类别交换不具备倾向性，即a类patch引入b类样本，使得b类概率降低，a类及其他类均升高，而非仅a类升高&其他类不变；V3方法完美解决。
- 2024-12-16: 在Exchange类中，增加dir和save_flag，用于保存完整attribute_res结果，用于V1、V2、V3的比较。
- 2024-12-16: 补充Mask和Scale的attribute代码。
## 代码运行
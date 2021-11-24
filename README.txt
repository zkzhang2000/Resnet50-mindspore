训练数据集路径为data/new_train
测试数据集路径为data/new_test


训练命令:python resnet50.py [--checkpoint (参数文件位置)]
测试命令:python eval.py [--checkpoint(参数文件位置,如何不设置默认为./this.ckpt)]


代码结构目录
├── resnet50
    ├── data
    │   ├── new_test			//test数据集
    │   ├── new_train		//train数据集
    ├── resnet50.py			//训练并给出评价
    ├── eval.py			//评价
    ├──resnet-ai_3-100_243.ckpt  	//参数文件


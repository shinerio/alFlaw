# 天池图像比赛Baseline分享
[2018广东工业智造大数据创新大赛——智能算法赛](https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.11165320.5678.1.54114443WSKVPP&raceId=231682)，未调参情况下线上`0.921`
---
## 运行代码前，需要将图片放在data目录下，目录树如下：
    
	|--data_center/alFlaw
		|--guangdong_round1_train1_20180903
		|--guangdong_round1_train2_20180916
		|--guangdong_round1_test_a_20180916
    

---
## 代码运行方式：
    
    更改config.person = "your name"，程序运行期间产生的所有日志，模型，结果文件都会存在data_center/alFlaw/yourname文件夹下，跑完的结果文件会自动移到archive
    参考文件目录
    |--data_center/alFlaw/rui.zhang
           |--archive 运行完train后，结果文件会自动移到该目录下
           |--label.csv  12类分类label
           |--binary_label.csv  2分类label
           |--result   test后结果文件存放
           |--summary.txt  历次运行结果统计
           |--num_classes_time 运行期间缓存文件，运行完会移动到archive文件夹下，程序异常中断不会移动，可以自行选择保留或删除 
       
    生成label
	python demo/demo_generate_label.py
    训练，验证，测试
	python demo/demo_train.py  
    验证
    python demo/demo_validate.py
    现场测试结果生成
    python demo/demo_predict_result.py
    
---
## 程序说明
clone自https://github.com/Herbert95/tianchi_lvcai
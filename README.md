

运行rl_trainer\main.py --max_train_steps=xx 后，模型存在model里。

把其中一个放到agent\rl里，然后改submission.py里的文件名，之后就能用evaluation_local.py了。

运行run_log.py就能得到跑动画需要的json，存在logs里。


更新：可以使用参数--load_model 来读取最新的model，继续训练了，重新训练请手动删除model\env下所有模型，否则会替换同名模型

可调参数有：lr,gamma,batch_size,target_update_freq,tau,use_lr_decay
	推测其中较敏感的数值有lr,gamma,tau

P.S.如果模型保存失败，很可能是路径有中文的原因


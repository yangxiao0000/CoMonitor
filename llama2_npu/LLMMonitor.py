# 兼容ClossalAI的LLM训练监控工具
# 团队名称：今天你科研了吗
import datetime
import json
import torch
import torch.nn as nn
import torch.distributed as dist
import csv
import os
import numpy as np
# import torch_npu

class Error(Exception):
    """
    注册hook失败报错
    """
    def __init__(self, message):
        self.message = message
        print(self.message)

def save_dict_list(data, format, filepath):
    """
    将字典组成的列表按指定格式保存到文件
    参数:
        data (list[dict]): 要保存的字典列表数据
        format (str): 文件格式，支持 'json', 'yaml', 'csv', 'txt'
        filepath (str): 目标文件完整路径（如：/data/output.json）
    异常:
        ValueError: 当传入不支持的格式时抛出
        ImportError: 当需要PyYAML库但未安装时抛出
    """
    # 验证支持的格式
    supported_formats = ['json', 'yaml', 'csv', 'txt']
    if format not in supported_formats:
        raise ValueError(f"不支持的格式: {format}。支持格式: {', '.join(supported_formats)}")

    # 自动创建目录
    os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None

    # 按格式处理数据
    if format == 'json':
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    
    elif format == 'yaml':
        try:
            import yaml
        except ImportError:
            raise ImportError("使用YAML格式需要PyYAML库，请执行: pip install PyYAML")
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.safe_dump_all(data, f, default_flow_style=False, allow_unicode=True)
    
    elif format == 'csv':
        # 自动收集所有字段名
        fieldnames = list({key for item in data for key in item.keys()})
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for item in data:
                # 将复杂值转为字符串
                processed = {
                    k: str(v) if not isinstance(v, (str, int, float, bool)) else v 
                    for k, v in item.items()
                }
                writer.writerow(processed)
    
    elif format == 'txt':
        with open(filepath, 'w', encoding='utf-8') as f:
            for i, item in enumerate(data, 1):
                f.write(f"===== 条目 {i} =====\n")
                for key, value in item.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")  # 条目间空行分隔

class LLMMonitor:
    """
    Monitor监控工具
    可以收集聚合前和聚合后的梯度
    """
    def __init__(self, config_file_path="monitor.json"):
        #读取用户自定义配置
        with open('config.json') as file:
            self.config=json.load(file)
        # print(self.config)       
        self.targets=self.config['targets']
        self.format=self.config['format']
        self.mode=self.config['mode']
        self.ops=self._validate_ops(self.config['ops'])
        # print(self.targets)
        # print(self.format)
        # print(self.mode)
        # print(self.ops)
        # 实时获取时间戳
        # print(datetime.date.today())
        # print(datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
        time=datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        # 更新文件名
        self.out_dir_bef="statbef"+time+'.'+self.format
        self.out_dir_aft="stataft"+time+'.'+self.format
        # print(self.out_dir_bef)
        # print(self.out_dir_aft)     
        #初始化变量
        self.param_grad=[]
        #分布式
        self.rank=dist.get_rank() if dist.is_initialized() else 0
        # print(self.rank)
    def set_monitor(self,model):
        """
        调用_register_hooks
        注册hook,实时收集各rank梯度
        """
        try:
            self._register_hooks(model)
            print("set monitor successfully!")
        except:
            raise Error("somthing went wrong when set monitor!")

    def mostep(self, model):
        """
        单步监控,根据hook收集的各rank的梯度计算聚合前梯度统计值并保存
        然后根据聚合后的梯度计算梯度统计值并保存
        """
        statbef=[]
        stataft=[]
        for per_param_grad in self.param_grad:
            per_stat_record={}
            per_stat_record['param_name']=per_param_grad['param_name']
            per_stat_record['rank']=per_param_grad['rank']
            for op in self.ops:
                try:
                    per_stat_record[op] = getattr(per_param_grad['grad'], op)().cpu().item()
                except RuntimeError as e:
                    per_stat_record[op] = float("nan")
                    print(f"计算{op}时出错 @ {param_name}: {str(e)}")
            # 使用环状缓冲区避免内存泄漏
            if len(statbef) > 1000:  # 保持最近1000条记录
                statbef.pop(0)
            statbef.append(per_stat_record)
        self.param_grad=[]
        #保存聚合前梯度统计值
        save_dict_list(statbef,self.format,self.out_dir_bef)
        #如果启动分布式训练，则计算聚合后的梯度统计值并保存
        if dist.is_initialized():
            for (param_name, param) in model.named_parameters():
                if ((param_name in self.targets) or (self.targets is not None))and param.requires_grad:
                    per_stat_record={}
                    per_stat_record['param_name']=param_name
                    #根据用户选项计算梯度统计值
                    for op in self.ops:
                        per_stat_record[op] = getattr(param.grad, op)().cpu().item()
                                # 使用环状缓冲区避免内存泄漏
                    if len(stataft) > 1000:  # 保持最近1000条记录
                        stataft.pop(0)
                    stataft.append(per_stat_record)
            save_dict_list(stataft,self.format,self.out_dir_aft)
            
    def _validate_ops(self, ops):
        """
        验证统计操作的有效性
        """
        # 验证统计操作的有效性
        valid_ops = []
        tensor = torch.empty(1)  # 用于方法验证的虚拟张量
        for op in ops:
            if not hasattr(tensor, op):
                raise ValueError(f"无效统计方法: {op} (非torch.Tensor方法)")
            if not callable(getattr(tensor, op)):
                raise ValueError(f"无效统计方法: {op} (不是可调用方法)")
            valid_ops.append(op)
        return valid_ops
    
    def _register_hooks(self,model):
        """
        为目标参数注册hook
        """
        # # 遍历模型的所有参数  
        for param_name, param in model.named_parameters():
            # 判断逻辑: (监控所有目标为None) 或 (参数在目标集合中) 且需要梯度
            if ((param_name in self.targets) or (self.targets is not None)) and param.requires_grad:
                param.register_hook(
                    lambda grad, param_name=parame_name: self.param_grad.append(
                        {'param_name':param_name,'rank':self.rank,'grad':grad}
                    )
                )

##########测试##################################
#用户接口
#json文件的读取
# import json
# with open('config.json') as file:
#     # 加载JSON数据
#     data = json.load(file)
# print(data)

#实时显示时间
# import datetime
# print(datetime.date.today())

#分布式
# Monitor=LLMMonitor()

# #register_hook
# import torch
# v = torch.tensor([0., 0., 0.], requires_grad=True)
# h = v.register_hook(lambda grad: grad * 2)  
# v.backward(torch.tensor([1., 2., 3.]))
# print(v.grad)
# h.remove()

#测试失败提醒
# raise Error("somthing went wrong when set monitor!")

#梯度统计值
# op="sum"
# grad=torch.tensor([1,2,3,4,5])
# print(float(getattr(grad, op)().item()))
##########测试##################################
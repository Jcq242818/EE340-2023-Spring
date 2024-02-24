"""argparse库的一些测试"""
# 导入库
import argparse

"""测试1"""
# 1. 定义命令行解析器对象
parser = argparse.ArgumentParser(description='Demo of argparse')

# 2. 添加命令行参数
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch', type=int, default=4)

# 3. 从命令行中结构化解析参数
args = parser.parse_args()
# print(args)
epochs = args.epochs
batch = args.batch
print('show {}  {}'.format(epochs, batch))

"""测试2"""
parser = argparse.ArgumentParser(description='Process some integers.')  # 构造一个命令行参数处理器的实例

parser.add_argument('integers', metavar='N', type=int, nargs="+", help='an integer.')  # 添加第一个参数
parser.add_argument('--sum', dest='accumulate', action='store_const', const=sum, default=max, help='sum.')  # 添加第二个参数
args = parser.parse_args()
print(args.accumulate(args.integers))  # 注意这里的accumulate和integers都是那两个参数各自的属性 dest ="accumulate"和"integers",

"""测试3"""
parser = argparse.ArgumentParser(description='LeNet')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--batch-size', '-b', default=256, type=int, help='Batches')
args = parser.parse_args()
print(args.lr)
print(args.batch_size)

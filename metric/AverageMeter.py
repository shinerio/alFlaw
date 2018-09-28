# 用于计算精度和时间的变化
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# 计算top K准确率
def accuracy(y_pred, y_actual, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    # for prob_threshold in np.arange(0, 1, 0.01):
    pred_count = y_actual.size(0)
    pred_correct_count = 0
    prob, pred = y_pred.topk(maxk, 1, True, True)
    # prob = np.where(prob > prob_threshold, prob, 0)
    for j in range(pred.size(0)):
        if int(y_actual[j]) == int(pred[j]):
            pred_correct_count += 1
    if pred_count == 0:
        final_acc = 0
    else:
        final_acc = pred_correct_count / pred_count
    return final_acc * 100, pred_count

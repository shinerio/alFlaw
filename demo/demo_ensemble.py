from collections import OrderedDict, Counter


def merge_multi_classes(base_dir, result_list, decider=None):
    merge_results = OrderedDict()

    for result in result_list:
        path = base_dir + "/" + result
        with open(path, 'r') as f:
            for line in f:
                data = line.replace('\n', '').split(',')
                if data[0] not in merge_results:
                    merge_results[data[0]] = [data[1]]
                else:
                    merge_results[data[0]] += [data[1]]
    return merge_results


def export_multi_classes_merged(merge_results, num_member):
    counter = [0 for i in range(num_member//2+1)]
    with open(base_dir + "/ensemble.csv", 'w') as f:
        for key in merge_results:
            most = Counter(merge_results[key]).most_common(1)[0]
            label = most[0]
            num = most[1]
            counter[num-num_member//2-1] += 1
            if num != num_member:
                print("conflicts exists, all member are {}".format(merge_results[key]))
                if num_member % 2 == 0 and num == num_member/2: # 偶数平票
                    most = Counter(merge_results[key]).most_common(2)
                    label1 = most[0][0]
                    label2 = most[0][0]
                    label = label1 if label1 == merge_results[key][decider] else (label2 if label2 == merge_results[key][decider] else label1)
            f.write("{},{}\n".format(key, label))
    print("merge over, the number of members of final results made by are {}".format(counter))


def ensemble_binary_multi_classes(binary, merge_results):
    with open(base_dir + "/ensemble.csv", 'w') as res:
        with open(binary, 'r') as f:
            for line in f:
                data = line.replace('\n', '').split(',')
                image = data[0]
                label = data[1]
                if label == "norm":
                    if "norm" != merge_results[image][2]:
                        print("error")
                    res.write("{},{}\n".format(image, label))
                else:
                    num_member = len(merge_results[image])
                    most = Counter(merge_results[image]).most_common(1)[0]
                    if merge_results[image][2]!="norm":
                        res.write("{},{}\n".format(image, merge_results[image][2]))
                        continue
                    print("error")
                    # 找到预测不是Norm的member, 如果都是，那就选最后一个
                    condidata = merge_results[image][-1]
                    for index in range(num_member - 1):
                        if merge_results[image][index] != "norm":
                            condidata = merge_results[image][index]
                    most_like_label = most[0] if most[0] != "norm" else condidata
                    num = most[1]
                    if num != num_member:
                        print("conflicts exists, all member are {}".format(merge_results[image]))
                        if num_member % 2 == 0 and num == num_member / 2:  # 偶数平票
                            most = Counter(merge_results[image]).most_common(2)
                            label1 = most[0][0]
                            label2 = most[0][0]
                            most_like_label = label1 if label1 !="norm" and label1 == merge_results[image][decider] else (
                                label2 if label2 !="norm" and label2 == merge_results[image][decider] else condidata)
                    res.write("{},{}\n".format(image, most_like_label))


def transform_mutil2binary():
    pass


if __name__ == '__main__':
    base_dir = '/home/messor/data_center/alFlaw/rui.zhang/result'
    result_list = ['submission_90.42.csv',
                   'submission_9105.csv',
                   'submission_938.csv',
                   'submission_all_in_9146.csv']
    binary_result = base_dir+"/binary_submission.csv"
    if len(result_list) % 2 == 0: # 偶数平票，最高者1.5票
        decider = 2
    else:
        decider = None
    merge_results = merge_multi_classes(base_dir, result_list, decider)
    # export_multi_classes_merged(merge_results, len(result_list))
    ensemble_binary_multi_classes(binary_result, merge_results)

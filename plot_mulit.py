import matplotlib.pylab as plt
import pandas as pd
import numpy as np
from data.get_data_20 import main_get_data


pd.set_option('display.float_format', lambda x1: '%.6f' % x1)  # pandas
np.set_printoptions(precision=6, suppress=True)  # numpy
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

# layout1 = (1, 2)
# data_train_ax = plt.subplot2grid(layout1, (0, 0))
# data_diff_ax = plt.subplot2grid(layout1, (0, 1))
font = {'family': 'SimSun',
        'weight': 'normal',  # normal  bold
        'size': '12'}
plt.rc('font', **font)


def plot_2(name_list, data_list, label):
    data_list_d = []
    plt.rcParams['figure.figsize'] = (12.0, 8.0)
    for i in range(len(data_list[0])):
        tmp_data = []
        for each_data in data_list:
            tmp_data.append(each_data[i])
        data_list_d.append(tmp_data)

    # num_list = data_list[0]
    # # num_list2 = [10, 12, 18, 26]
    # num_list2 = data_list[1]
    patterns = ('----', '////', '\\\\\\', '....', '^^oo', '* * *', '-', 'x', '\\', '/', '+', 'O')
    x = list(range(len(data_list_d[0])))
    x = [1, 3, 5, 7]
    # font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}
    """
    宋体     SimSun
    黑体     SimHei
    微软雅黑     Microsoft YaHei
    微软正黑体     Microsoft JhengHei
    新宋体     NSimSun
    新细明体     PMingLiU
    细明体     MingLiU
    标楷体     DFKai-SB
    仿宋     FangSong
    楷体     KaiTi
    隶书：LiSu
    幼圆：YouYuan
    华文细黑：STXihei
    华文楷体：STKaiti
    华文宋体：STSong
    华文中宋：STZhongsong
    华文仿宋：STFangsong
    方正舒体：FZShuTi
    方正姚体：FZYaoti
    华文彩云：STCaiyun
    华文琥珀：STHupo
    华文隶书：STLiti
    华文行楷：STXingkai
    华文新魏：STXinwei
    """
    font = {'family': 'SimSun',
            'weight': 'normal',  # normal  bold
            'size': '12'}
    plt.rc('font', **font)
    total_width, n = 1.6, len(data_list_d)
    width = total_width / n
    mid = int(len(data_list_d)/2)
    plt.bar(x, data_list_d[0],
            hatch=patterns[0],
            width=width,
            label=name_list[0],
            color='white',
            edgecolor='k')

    for j in range(1, len(data_list_d)):
        for i in range(len(x)):
            x[i] += width
        if j != mid:
            plt.bar(x,
                    data_list_d[j],
                    hatch=patterns[j],
                    width=width,
                    label=name_list[j],
                    color='white',
                    edgecolor='k')
        else:
            plt.bar(x, data_list_d[j],
                    hatch=patterns[j],
                    width=width,
                    tick_label=label,
                    label=name_list[j],
                    color='white',
                    edgecolor='k')
    # plt.legend(loc='String or Number',  bbox_to_anchor=(num1, num2))
    plt.legend(loc='upper right')
    """
    String          Number
    upper right     1
    upper left      2
    lower left      3
    lower right     4
    right           5
    center left     6
    center right    7
    lower center    8
    upper center    9
    center          10
    """
    plt.show()


# 显示高度
def autolabel(rects):

    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2. - 0.25, 1.02*height, "{:.3f}".format(height))


def plot_1(name_list, num_list, num, type_):

    if type_ == 'MRE':
        type_n = 'MRE'
        num_list = [n * 100 for n in num_list]
    else:
        type_n = 'RMSE'
    # num_list = [n * 10 for n in num_list]
    type_n = num + '-' + type_n

    # name_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    # num_list = [33, 44, 53, 16, 11, 17, 17, 10]
    size = 9
    plt.xticks(np.arange(len(name_list)), name_list, rotation=0, fontsize=size)
    plt.yticks(fontsize=size)
    # plt.title(type_n)
    # autolabel(plt.bar(range(len(num_list)), num_list, width=0.5, color='gray'))
    autolabel(plt.bar(range(len(num_list)), num_list, width=0.5, color='rgb'))
    # autolabel(plt.bar(range(len(num_list)), num_list, color='rgb', tick_label=name_list))
    """
    'aliceblue':            '#F0F8FF',
    'antiquewhite':         '#FAEBD7',
    'aqua':                 '#00FFFF',
    'aquamarine':           '#7FFFD4',
    'azure':                '#F0FFFF',
    'beige':                '#F5F5DC',
    'bisque':               '#FFE4C4',
    'black':                '#000000',
    'blanchedalmond':       '#FFEBCD',
    'blue':                 '#0000FF',
    'blueviolet':           '#8A2BE2',
    'brown':                '#A52A2A',
    'burlywood':            '#DEB887',
    'cadetblue':            '#5F9EA0',
    'chartreuse':           '#7FFF00',
    'chocolate':            '#D2691E',
    'coral':                '#FF7F50',
    'cornflowerblue':       '#6495ED',
    'cornsilk':             '#FFF8DC',
    'crimson':              '#DC143C',
    'cyan':                 '#00FFFF',
    'darkblue':             '#00008B',
    'darkcyan':             '#008B8B',
    'darkgoldenrod':        '#B8860B',
    'darkgray':             '#A9A9A9',
    'darkgreen':            '#006400',
    'darkkhaki':            '#BDB76B',
    'darkmagenta':          '#8B008B',
    'darkolivegreen':       '#556B2F',
    'darkorange':           '#FF8C00',
    'darkorchid':           '#9932CC',
    'darkred':              '#8B0000',
    'darksalmon':           '#E9967A',
    'darkseagreen':         '#8FBC8F',
    'darkslateblue':        '#483D8B',
    'darkslategray':        '#2F4F4F',
    'darkturquoise':        '#00CED1',
    'darkviolet':           '#9400D3',
    'deeppink':             '#FF1493',
    'deepskyblue':          '#00BFFF',
    'dimgray':              '#696969',
    'dodgerblue':           '#1E90FF',
    'firebrick':            '#B22222',
    'floralwhite':          '#FFFAF0',
    'forestgreen':          '#228B22',
    'fuchsia':              '#FF00FF',
    'gainsboro':            '#DCDCDC',
    'ghostwhite':           '#F8F8FF',
    'gold':                 '#FFD700',
    'goldenrod':            '#DAA520',
    'gray':                 '#808080',
    'green':                '#008000',
    'greenyellow':          '#ADFF2F',
    'honeydew':             '#F0FFF0',
    'hotpink':              '#FF69B4',
    'indianred':            '#CD5C5C',
    'indigo':               '#4B0082',
    'ivory':                '#FFFFF0',
    'khaki':                '#F0E68C',
    'lavender':             '#E6E6FA',
    'lavenderblush':        '#FFF0F5',
    'lawngreen':            '#7CFC00',
    'lemonchiffon':         '#FFFACD',
    'lightblue':            '#ADD8E6',
    'lightcoral':           '#F08080',
    'lightcyan':            '#E0FFFF',
    'lightgoldenrodyellow': '#FAFAD2',
    'lightgreen':           '#90EE90',
    'lightgray':            '#D3D3D3',
    'lightpink':            '#FFB6C1',
    'lightsalmon':          '#FFA07A',
    'lightseagreen':        '#20B2AA',
    'lightskyblue':         '#87CEFA',
    'lightslategray':       '#778899',
    'lightsteelblue':       '#B0C4DE',
    'lightyellow':          '#FFFFE0',
    'lime':                 '#00FF00',
    'limegreen':            '#32CD32',
    'linen':                '#FAF0E6',
    'magenta':              '#FF00FF',
    'maroon':               '#800000',
    'mediumaquamarine':     '#66CDAA',
    'mediumblue':           '#0000CD',
    'mediumorchid':         '#BA55D3',
    'mediumpurple':         '#9370DB',
    'mediumseagreen':       '#3CB371',
    'mediumslateblue':      '#7B68EE',
    'mediumspringgreen':    '#00FA9A',
    'mediumturquoise':      '#48D1CC',
    'mediumvioletred':      '#C71585',
    'midnightblue':         '#191970',
    'mintcream':            '#F5FFFA',
    'mistyrose':            '#FFE4E1',
    'moccasin':             '#FFE4B5',
    'navajowhite':          '#FFDEAD',
    'navy':                 '#000080',
    'oldlace':              '#FDF5E6',
    'olive':                '#808000',
    'olivedrab':            '#6B8E23',
    'orange':               '#FFA500',
    'orangered':            '#FF4500',
    'orchid':               '#DA70D6',
    'palegoldenrod':        '#EEE8AA',
    'palegreen':            '#98FB98',
    'paleturquoise':        '#AFEEEE',
    'palevioletred':        '#DB7093',
    'papayawhip':           '#FFEFD5',
    'peachpuff':            '#FFDAB9',
    'peru':                 '#CD853F',
    'pink':                 '#FFC0CB',
    'plum':                 '#DDA0DD',
    'powderblue':           '#B0E0E6',
    'purple':               '#800080',
    'red':                  '#FF0000',
    'rosybrown':            '#BC8F8F',
    'royalblue':            '#4169E1',
    'saddlebrown':          '#8B4513',
    'salmon':               '#FA8072',
    'sandybrown':           '#FAA460',
    'seagreen':             '#2E8B57',
    'seashell':             '#FFF5EE',
    'sienna':               '#A0522D',
    'silver':               '#C0C0C0',
    'skyblue':              '#87CEEB',
    'slateblue':            '#6A5ACD',
    'slategray':            '#708090',
    'snow':                 '#FFFAFA',
    'springgreen':          '#00FF7F',
    'steelblue':            '#4682B4',
    'tan':                  '#D2B48C',
    'teal':                 '#008080',
    'thistle':              '#D8BFD8',
    'tomato':               '#FF6347',
    'turquoise':            '#40E0D0',
    'violet':               '#EE82EE',
    'wheat':                '#F5DEB3',
    'white':                '#FFFFFF',
    'whitesmoke':           '#F5F5F5',
    'yellow':               '#FFFF00',
    'yellowgreen':          '#9ACD32'}
    """

    """
    '-'       solid line style
    '--'      dashed line style
    '-.'      dash-dot line style
    ':'       dotted line style
    """

    """
    '.'       point marker
    ','       pixel marker
    'o'       circle marker
    'v'       triangle_down marker
    '^'       triangle_up marker
    '<'       triangle_left marker
    '>'       triangle_right marker
    '1'       tri_down marker
    '2'       tri_up marker
    '3'       tri_left marker
    '4'       tri_right marker
    's'       square marker
    'p'       pentagon marker
    '*'       star marker
    'h'       hexagon1 marker
    'H'       hexagon2 marker
    '+'       plus marker
    'x'       x marker
    'D'       diamond marker
    'd'       thin_diamond marker
    '|'       vline marker
    '_'       hline marker
    """
    # plt.legend()
    plt.show()


def get_result(path):
    data = pd.read_csv(path, index_col=0, parse_dates=[0])
    # date
    # close_000017	close_000017_AFSA_GA_RBF	close_000017_AFSA_RBF	close_000017_GSA_RBF	close_000017_RBF	close_000017_ARIMA	close_000017_LSTM
    # close_300730	close_300730_AFSA_GA_RBF	close_300730_AFSA_RBF	close_300730_GSA_RBF	close_300730_RBF	close_300730_ARIMA	close_300730_LSTM
    # close_300144	close_300144_AFSA_GA_RBF	close_300144_AFSA_RBF	close_300144_GSA_RBF	close_300144_RBF	close_300144_ARIMA	close_300144_LSTM
    # close_002628	close_002628_AFSA_GA_RBF	close_002628_AFSA_RBF	close_002628_GSA_RBF	close_002628_RBF	close_002628_ARIMA	close_002628_LSTM
    # print(data['close_000017'])

    def get_mre(data_o, data_f):
        return (abs(data_f - data_o)/data_o).sum()/(data_o.shape[0])

    def get_rmse(data_o, data_f):

        return np.sqrt(((data_f - data_o)*(data_f - data_o)).sum()/(data_o.shape[0]))
        # return (abs(data_f - data_o)/data_o).sum()/data_o.shape[0]

    txt = ['AFSA_GA_RBF', 'AFSA_RBF', 'GSA_RBF', 'RBF', 'ARIMA', 'LSTM']

    def plot_mae_mrse(data_list, lable_list, type_):

        plt.plot(data_list[0], 'k-')
        # plt.plot(data_list[1], 'k-.')
        # plt.plot(data_list[2], 'k:')
        # plt.plot(data_list[3], 'k--')
        size = 12
        plt.legend(lable_list, fontsize=size + 5)
        # plt.legend(lable_list, fontsize=size + 5, loc='upper right')
        plt.xticks(np.arange(len(txt)), txt, rotation=0, fontsize=size)
        plt.yticks(fontsize=size)
        font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}
        plt.xlabel("modle", font2)
        plt.ylabel(type_, font2)
        # plt.title("GSA+AFSA")
        # plt.legend(loc='upper right')
        plt.show()

    def get_mre_rmse(number):
        # txt = ['AFSA_GA_RBF', 'AFSA_RBF', 'GSA_RBF', 'RBF', 'ARIMA', 'LSTM']
        index_o = 'close_' + number
        rmse = []
        mre = []
        for i in txt:
            rmse.append(get_rmse(data[index_o], data['close_' + str(number) + '_' + i]))
            mre.append(get_mre(data[index_o], data['close_' + str(number) + '_' + i]))

        return rmse, mre
    """
    创业板股票：科创信息（300730）、宋城演艺（300144）；中小板股票：成都路桥（002628））
    """
    # label = ['000017', '300730', '002628', '300144']
    label = ['000017', '600519', '300730', '603991']

    # label = ['000017']
    # num_list = ['002628']
    rmse_list, mre_list = [], []
    for each_num in label:
        rmse_tmp, mre_tmp = get_mre_rmse(each_num)
        rmse_list.append(rmse_tmp)
        mre_list.append(mre_tmp)
        # print(rmse_list)
        # print(mre_list)
    # label = ['深中华(000017)', '科创信息(300730)', '成都路桥(002628)', '宋城演艺(300144)']
    label = ['深中华(000017)', '贵州茅台(600519)', '科创信息(300730)', '至正股份(603991)']
    # print(mre_list[0])
    # print(mre_list[1])
    plot_2(txt, rmse_list, label)
    plot_2(txt, mre_list, label)
    # plot_mae_mrse(mre_list, label, 'MRE')
    # plot_mae_mrse(rmse_list, label, 'RMSE')
    plt.show()


if __name__ == '__main__':
    # path_r = 'E:\\Document\\python\\deep_python\\Optimization_algorithm\\NET_work\\预测结果过表.csv'
    path_r = 'E:\\git\\RBF_AFSA_GSA\\data\\result_data.csv'

    # 保存数据
    # main_get_data(path_r)

    print('__________11111111111111____________')
    get_result(path_r)
    # plot_1()
    # plot_2()

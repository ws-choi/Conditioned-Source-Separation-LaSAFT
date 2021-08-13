import shutil
import wget


def get_lasaft_large(seed, save_path):
    assert seed in [2019, 2020, 2021, '2019', '2020', '2021']
    url = 'http://intelligence.korea.ac.kr/assets/lasaft_large_' + str(seed) + '.ckpt'
    wget.download(url)

    print('no cached checkpoint found.\nautomatic download!')
    shutil.move('lasaft_large_' + str(seed) + '.ckpt', save_path)

    print('successfully downloaded the pretrained model.')

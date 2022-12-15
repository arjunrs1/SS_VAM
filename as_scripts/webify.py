import os
import datetime
import sys
import shutil
import glob
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--imgsize', type=int, default=100)
    parser.add_argument('--num', type=int, default=10000)

    args = parser.parse_args()
    return args


def create_html(path, f, name):
    path = os.path.join('/vision/asomaya1/vam_mod/audio_synthesis_main', path)
    f.write('''<div align = "center">''')

    audio_sec = """
<h3>{}</h3>
<table>
<tbody>
<tr>
<th>Index #</th>
<th>Input/prediction/target</th>
<th>Source Image</th>
<th>Target Image</th>
<th>Monaural Output</th>
</tr>\n
    """.format(name)
    f.write(audio_sec)
    total_num_file = len(os.listdir(path)) // 9

    for i in range(total_num_file):
        f.write('''<tr>\n''')
        f.write('''<td> audio #{} </td>'''.format(str(i)))
        f.write('\n')

        # audio
        f.write('''<td>''')
        for audio_type in ['input', 'pred', 'tgt']:
            f.write('''<audio controls=""><source src='{}' type="audio/{}"></audio> <br>'''.format(
                os.path.join(path, f'{i}-{audio_type}.wav'), 'wav'))
            f.write('\n')
        f.write('''</td>''')

        # images
        for img_type in ['src-rgb', 'tgt-rgb']:
            f.write('''<td> <div align = "center"><img src='{}' style="zoom:{}%" /></div> </td>'''.format(
                os.path.join(path, f'{i}-{img_type}.png'), 70))
            f.write('\n')

        # plots
        f.write('''<td> <div align = "center"><img src='{}' style="zoom:{}%" /></div> </td>'''.format(
            os.path.join(path, f'{i}-plot.png'), 55))
        f.write('\n')

        f.write('''</tr>\n''')

    f.write('''</tbody></table>\n''')
    f.write('''</div>\n''')

    # for video
    # '''<td> <div align = "center"><video id='{}' controls height='200'><source src="{}" type="video/{}" preload = "none"></video> <p>Speed: <input type="text" size="5" value="1" oninput="document.getElementById('{}').playbackRate = parseFloat(event.target.value);"></p></div> </td>'''.format(sample, sample, sample.split('.')[-1], sample))


def webify(path, version):
    html_file = os.path.join(path, 'index.html')
    f = open(html_file, 'wt')

    # head
    # <link rel="stylesheet" type="text/css" title="Cool stylesheet" href="style.css">
    head = """<!DOCTYPE html>
<html lang="en"><head><meta http-equiv="Content-Type" content="text/html; charset=UTF-8">

<meta name="viewport" content="width=device-width, initial-scale=1.0">

</head>
    """
    f.write(head)

    intro_sec = '''
<body data-gr-c-s-loaded="true">

'''
    f.write(intro_sec)
    create_html(path, f, version)
    f.write('''</body>\n''')
    f.write('''</html>\n''')
    f.close()


if __name__ == "__main__":
    args = parse_args()
    webify(args)
    print('Webify Succeed!')

# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from argparse import ArgumentParser
from torch.cuda import device_count
from torch.multiprocessing import spawn

from learner import train, train_distributed
from params import params


def _get_free_port():
  import socketserver
  with socketserver.TCPServer(('localhost', 0), None) as s:
    return s.server_address[1]


def main(args):
  if args.train_reverb_vocoder:
    assert not args.use_frozen_dereverb

  replica_count = device_count()
  if replica_count > 1:
    if params.batch_size % replica_count != 0:
      raise ValueError(f'Batch size {params.batch_size} is not evenly divisble by # GPUs {replica_count}.')
    params.batch_size = params.batch_size // replica_count
    port = _get_free_port()
    spawn(train_distributed, args=(replica_count, port, args, params), nprocs=replica_count, join=True)
  else:
    train(args, params)


if __name__ == '__main__':
  parser = ArgumentParser(description='train (or resume training) a DiffWave model')
  parser.add_argument('--model-dir',
      help='directory in whichs to store model checkpoints and training logs')
  parser.add_argument('--dataset', default='soundspaces',
      help='which dataset to use for training')
  parser.add_argument('--max-steps', default=None, type=int,
      help='maximum number of training steps')
  parser.add_argument('--split', default='train',
      help='data split to be used for training')
  parser.add_argument('--use-dereverb-sspeech', action='store_true', default=False,
      help='dereverberation mode')
  parser.add_argument('--use-pretrained', action='store_true', default=False,
      help='use pretrained reconstruction DiffWave model')
  parser.add_argument('--use-frozen-dereverb', action='store_true', default=False,
      help='use WaveNet dereverberator front-end')
  parser.add_argument('--test', action='store_true', default=False,
      help='evaluate model on test set')
  parser.add_argument('--test-split', default='test-unseen',
      help='data split to be used for testing')
  parser.add_argument('--fp16', action='store_true', default=False,
      help='use 16-bit floating point operations for training')
  parser.add_argument('--use-rt60-feature', action='store_true', default=False,
      help='use rt60 as visual feature')
  parser.add_argument('--train-reverb-vocoder', action='store_true', default=False,
      help='use reverb input and output')
  parser.add_argument('--visualize', action='store_true', default=False,
      help='generate visualization')
  parser.add_argument('--sample-rate', default=16000, type=int,
      help='audio sampling rate')
  main(parser.parse_args())

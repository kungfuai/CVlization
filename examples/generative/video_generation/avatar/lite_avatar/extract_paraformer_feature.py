#coding=utf-8
import torch,os,sys,time,codecs,argparse,logging
from pathlib import Path
from typing import Optional,Tuple,Union,Dict,Any,List
import numpy as np
from typeguard import check_argument_types
from funasr_local.modules.beam_search.beam_search import BeamSearchPara as BeamSearch
from funasr_local.modules.beam_search.beam_search import Hypothesis
from funasr_local.modules.scorers.ctc import CTCPrefixScorer
from funasr_local.modules.scorers.length_bonus import LengthBonus
from funasr_local.tasks.asr import ASRTaskParaformer as ASRTask
from funasr_local.tasks.lm import LMTask
from funasr_local.text.build_tokenizer import build_tokenizer
from funasr_local.text.token_id_converter import TokenIDConverter
from funasr_local.torch_utils.device_funcs import to_device
from funasr_local.torch_utils.set_all_random_seed import set_all_random_seed
from funasr_local.utils import asr_utils, wav_utils, postprocess_utils
from funasr_local.models.frontend.wav_frontend import WavFrontend
import soundfile
import torch.nn.functional as F
import os
import time
global_asr_language: str = 'zh-cn'
global_sample_rate: Union[int, Dict[Any, int]] = {
    'audio_fs': 16000,
    'model_fs': 16000
}

def linear_interpolation(features, output_len=None):
    features = features.transpose(1, 2)
    output_features = F.interpolate(features,size=output_len,align_corners=True,mode='linear')
    return output_features.transpose(1, 2)

class Speech2Text:
    def __init__(
            self,
            asr_train_config: Union[Path, str] = None,
            asr_model_file: Union[Path, str] = None,
            cmvn_file: Union[Path, str] = None,
            lm_train_config: Union[Path, str] = None,
            lm_file: Union[Path, str] = None,
            token_type: str = None,
            bpemodel: str = None,
            device: str = "cpu",
            maxlenratio: float = 0.0,
            minlenratio: float = 0.0,
            dtype: str = "float32",
            beam_size: int = 20,
            ctc_weight: float = 0.5,
            lm_weight: float = 1.0,
            ngram_weight: float = 0.9,
            penalty: float = 0.0,
            nbest: int = 1,
            frontend_conf: dict = None,
            **kwargs,
    ):
        assert check_argument_types()

        # 1. Build ASR model
        scorers = {}
        asr_model, asr_train_args = ASRTask.build_model_from_file(
            asr_train_config, asr_model_file, cmvn_file, device=device
        )
        if asr_model.frontend is None and frontend_conf is not None:
            frontend = WavFrontend(**frontend_conf)
            asr_model.frontend = frontend
        # logging.info("asr_model: {}".format(asr_model))
        # logging.info("asr_train_args: {}".format(asr_train_args))
        asr_model.to(dtype=getattr(torch, dtype)).eval()

        # ctc = CTCPrefixScorer(ctc=asr_model.ctc, eos=asr_model.eos)
        # token_list = asr_model.token_list
        # scorers.update(
        #     ctc=ctc,
        #     length_bonus=LengthBonus(len(token_list)),
        # )

        # # 2. Build Language model
        # if lm_train_config is not None:
        #     lm, lm_train_args = LMTask.build_model_from_file(
        #         lm_train_config, lm_file, device=device
        #     )
        #     scorers["lm"] = lm.lm

        # # 3. Build ngram model
        # # ngram is not supported now
        # ngram = None
        # scorers["ngram"] = ngram

        # # 4. Build BeamSearch object
        # # transducer is not supported now
        # beam_search_transducer = None

        # weights = dict(
        #     decoder=1.0 - ctc_weight,
        #     ctc=ctc_weight,
        #     lm=lm_weight,
        #     ngram=ngram_weight,
        #     length_bonus=penalty,
        # )
        # beam_search = BeamSearch(
        #     beam_size=beam_size,
        #     weights=weights,
        #     scorers=scorers,
        #     sos=asr_model.sos,
        #     eos=asr_model.eos,
        #     vocab_size=len(token_list),
        #     token_list=token_list,
        #     pre_beam_score_key=None if ctc_weight == 1.0 else "full",
        # )
        # beam_search.to(device=device, dtype=getattr(torch, dtype)).eval()
        # for scorer in scorers.values():
        #     if isinstance(scorer, torch.nn.Module):
        #         scorer.to(device=device, dtype=getattr(torch, dtype)).eval()
        # # 5. [Optional] Build Text converter: e.g. bpe-sym -> Text
        # if token_type is None:
        #     token_type = asr_train_args.token_type
        # if bpemodel is None:
        #     bpemodel = asr_train_args.bpemodel

        # if token_type is None:
        #     tokenizer = None
        # elif token_type == "bpe":
        #     if bpemodel is not None:
        #         tokenizer = build_tokenizer(token_type=token_type, bpemodel=bpemodel)
        #     else:
        #         tokenizer = None
        # else:
        #     tokenizer = build_tokenizer(token_type=token_type)
        # converter = TokenIDConverter(token_list=token_list)
        # # logging.info(f"Text tokenizer: {tokenizer}")
        self.asr_model = asr_model
        # self.asr_train_args = asr_train_args
        # self.converter = converter
        # self.tokenizer = tokenizer
        # has_lm = lm_weight == 0.0 or lm_file is None
        # if ctc_weight == 0.0 and has_lm:
        #     beam_search = None
        # self.beam_search = beam_search
        # self.beam_search_transducer = beam_search_transducer
        # self.maxlenratio = maxlenratio
        # self.minlenratio = minlenratio
        self.device = device
        # self.dtype = dtype
        # self.nbest = nbest

    @torch.no_grad()
    def __call__(
            self, speech: Union[torch.Tensor, np.ndarray], speech_lengths: Union[torch.Tensor, np.ndarray] = None, frame_cnt = None
    ):
        assert check_argument_types()
        # Input as audio signal
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)

        lfr_factor = max(1, (speech.size()[-1]//80)-1)
        batch = {"speech": speech, "speech_lengths": speech_lengths}
        # a. To device
        batch = to_device(batch, device=self.device)
        # b. Forward Encoder
        enc_out, enc_len = self.asr_model.encode(**batch)
        if isinstance(enc_out, tuple):
            enc = enc_out[0]
            hidden_states = enc_out[1]
        # print(enc.size())
        interp_enc = linear_interpolation(enc, frame_cnt)
        # print(interp_enc.size())
        
        
        
        interp_features = []
        for hid in hidden_states:
            interp_enc = linear_interpolation(hid, frame_cnt)
            interp_features.append(interp_enc[0].cpu().numpy())
        interp_features = np.asarray(interp_features).transpose(1,0,2)
        # print(interp_features.shape)
        '''
        
        # assert len(enc) == 1, len(enc)
        enc_len_batch_total = torch.sum(enc_len).item()

        predictor_outs = self.asr_model.calc_predictor(enc, enc_len)
        pre_acoustic_embeds, pre_token_length = predictor_outs[0], predictor_outs[1]
        pre_token_length = pre_token_length.round().long()
        decoder_outs = self.asr_model.cal_decoder_with_predictor(enc, enc_len, pre_acoustic_embeds, pre_token_length)
        decoder_out, ys_pad_lens = decoder_outs[0], decoder_outs[1]

        results = []
        b, n, d = decoder_out.size()
        for i in range(b):
            x = enc[i, :enc_len[i], :]
            am_scores = decoder_out[i, :pre_token_length[i], :]
            if self.beam_search is not None:
                nbest_hyps = self.beam_search(
                    x=x, am_scores=am_scores, maxlenratio=self.maxlenratio, minlenratio=self.minlenratio
                )
    
                nbest_hyps = nbest_hyps[: self.nbest]
            else:
                yseq = am_scores.argmax(dim=-1)
                score = am_scores.max(dim=-1)[0]
                score = torch.sum(score, dim=-1)
                # pad with mask tokens to ensure compatibility with sos/eos tokens
                yseq = torch.tensor(
                    [self.asr_model.sos] + yseq.tolist() + [self.asr_model.eos], device=yseq.device
                )
                nbest_hyps = [Hypothesis(yseq=yseq, score=score)]
                
            for hyp in nbest_hyps:
                assert isinstance(hyp, (Hypothesis)), type(hyp)
    
                # remove sos/eos and get results
                last_pos = -1
                if isinstance(hyp.yseq, list):
                    token_int = hyp.yseq[1:last_pos]
                else:
                    token_int = hyp.yseq[1:last_pos].tolist()
    
                # remove blank symbol id, which is assumed to be 0
                token_int = list(filter(lambda x: x != 0, token_int))
    
                # Change integer-ids to tokens
                token = self.converter.ids2tokens(token_int)
    
                if self.tokenizer is not None:
                    text = self.tokenizer.tokens2text(token)
                else:
                    text = None
    
                results.append((text, token, token_int, hyp, enc_len_batch_total, lfr_factor))
        # assert check_return_type(results)
        '''
        return interp_features

model_path="./weights/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/" #模型路径
model_type='pytorch'
ngpu=0
log_level='ERROR'

data_path_and_name_and_type=['speech','sound',model_path+'am.mvn']
asr_model_file=model_path+'model.pb'
cmvn_file=model_path+'am.mvn'
idx_text=''
sampled_ids='seq2seq/sampled_ids'
sampled_lengths='seq2seq/sampled_lengths'
lang='zh-cn'
code_base='funasr_local'
mode='paraformer'
fs={'audio_fs=16000','model_fs=16000'}
beam_size=1
penalty=0.0
maxlenratio=0.0
minlenratio=0.0
ctc_weight=0.0
lm_weight=0.0
asr_train_config=model_path+'config.yaml'
lm_file=model_path+'lm/lm.pb'
lm_train_config=model_path+'lm/lm.yaml'
batch_size=1
frontend_conf={'fs':16000,'win_length':400,'hop_length':160,'window':'hamming','n_mels': 80, 'lfr_m': 7, 'lfr_n': 6}
token_num_relax=None
decoding_ind=None
decoding_mode=None
num_workers=0
device='cpu' #GPU设置'device':'cuda' CPU设置'device':'cpu'
# device='cpu'
token_type: Optional[str] = None
key_file: Optional[str] = None
word_lm_train_config: Optional[str] = None
bpemodel: Optional[str] = None
allow_variable_data_keys: bool = False
streaming: bool = False
dtype: str = "float32"
ngram_weight: float = 0.9
nbest: int = 1
fs: Union[dict, int] = 16000

hop_length: int = 160
sr = 16000

if isinstance(data_path_and_name_and_type[0], Tuple):
    features_type: str = data_path_and_name_and_type[0][1]
elif isinstance(data_path_and_name_and_type[0], str):
    features_type: str = data_path_and_name_and_type[1]
else:
    raise NotImplementedError("unknown features type:{0}".format(data_path_and_name_and_type))
if features_type != 'sound':
    frontend_conf = None
    flag_modelscope = False
else:
    flag_modelscope = True
if frontend_conf is not None:
    if 'hop_length' in frontend_conf:
        hop_length = frontend_conf['hop_length']
set_all_random_seed(0)
# 2. Build speech2text
speech2text_kwargs = dict(
    asr_train_config=asr_train_config,
    asr_model_file=asr_model_file,
    cmvn_file=cmvn_file,
    lm_train_config=lm_train_config,
    lm_file=lm_file,
    token_type=token_type,
    bpemodel=bpemodel,
    device=device,
    maxlenratio=maxlenratio,
    minlenratio=minlenratio,
    dtype=dtype,
    beam_size=beam_size,
    ctc_weight=ctc_weight,
    lm_weight=lm_weight,
    ngram_weight=ngram_weight,
    penalty=penalty,
    nbest=nbest,
    frontend_conf=frontend_conf,
)
# print(speech2text_kwargs);input('')
speech2text = Speech2Text(**speech2text_kwargs)
# 3. Build data-iterator

def extract_para_feature(audio, frame_cnt):
    s = time.time()
    
    # results = speech2text(**batch)
    batch_ = {"speech": torch.tensor(np.array([audio],dtype=np.float32)), 'frame_cnt': frame_cnt, "speech_lengths": torch.tensor(np.array([len(audio)]))}
    # print('batch_',batch_ )#;input('')
    results = speech2text(**batch_)
    print('extract paraformer feature in {}ms'.format(round((time.time() - s),3)))
    return results


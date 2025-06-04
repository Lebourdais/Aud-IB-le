import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset

class CommonVoiceDataset(Dataset):
    def __init__(self,mode="train",samplerate=16000,length=2.0,label="gender"):
        super().__init__()
        dataset = load_dataset("mozilla-foundation/common_voice_11_0", "fr", split=mode)
        dataset = dataset.cast_column("audio", Audio(sampling_rate=samplerate))
        dataset = dataset.remove_columns(["sentence","up_votes","down_votes","age","accent","locale","segment"])
        print("Cleaning...",end="")
        dataset = dataset.filter(lambda example: example,input_columns=label)
        print("..",end="")
        dataset = dataset.map(self.to_index,input_columns=label)
        dataset.set_format(type='torch', columns=['audio', 'gender'])
        print("..OK")
        self.samplerate = samplerate
        self.set = dataset
        self.length=length
        self.sample_length=int(length*samplerate)
        
        
    def to_index(self,example):
        example = 1 if example=='female' else 0
        dicti={"gender":example}
        return dicti
    
    def __getitem__(self,idx):
        sample = self.set.__getitem__(idx)
        label = sample['gender']
        audio = sample['audio']['array']
        
        
        audio = F.pad(audio[:self.sample_length],(0,max(0,self.sample_length-audio.shape[0])), "constant", 0)
        return audio,label
    def __len__(self):
        return len(self.set)

class TimitDataset(Dataset):

    def __init__(self,mode="train",samplerate=16000,length=0.5,dataset_utterance=None):
        super().__init__()
        dataset = load_dataset('timit_asr', data_dir='/lium/raid01_b/mlebour/data/timit/TIMIT/',split=mode)
        #dataset = load_dataset("mozilla-foundation/common_voice_11_0", "fr", split=mode)
        self.mapping_phone = {
                'iy':0,
                'ih':1,
                'ix':1,
                'eh':2,
                'ae':3,
                'ax':4,
                'ah':4,
                'ax-h':4,
                'uw':5,
                'ux':5,
                'uh':6,
                'ao':7,
                'aa':7,
                'ey':8,
                'ay':9,
                'oy':10,
                'aw':11,
                'ow':12,
                'er':13,
                'axr':13,
                'l':14,
                'el':14,
                'r':15,
                'w':16,
                'y':17,
                'm':18,
                'em':18,
                'n':19,
                'en':19,
                'nx':19,
                'ng':20,
                'eng':20,
                'v':21,
                'f':22,
                'dh':23,
                'th':24,
                'z':25,
                's':26,
                'zh':27,
                'sh':27,
                'jh':28,
                'ch':29,
                'b':30,
                'p':31,
                'd':32,
                'dz':33,
                't':34,
                'g':35,
                'k':36,
                'hh':37,
                'hv':37,
        }
        
        dataset = dataset.cast_column("audio", Audio(sampling_rate=samplerate))
        
        dataset = dataset.remove_columns(["text","word_detail","dialect_region","sentence_type","speaker_id","id"])
        print("Cleaning...",end="")
        print("..",end="")
        
        phonemes = []
        for detail in dataset['phonetic_detail']:
            for p in detail['utterance']:
                if p not in phonemes:
                    phonemes.append(p)

        self.list_phone = ['iy','ih','eh','ae','ax','uw','uh','ao','ey','ay','oy','aw','ow','er','l','r','w','y','m','n','ng','v','f','dh','th','z','s','zh','jh','ch','b','p','d','dz','t','g','k','hh','sil']
        
        
        dataset = dataset.map(self.to_index,input_columns='phonetic_detail')
        
        dataset.set_format(type='torch', columns=['audio', 'phonetic_detail','start','stop'])
        
        print("..OK")
        self.samplerate = samplerate
        self.length=length
        self.sample_length=int(length*samplerate)
        print("Dataset preparation")
        self.phone = []
        for e in dataset:
            
            for label,start,stop in zip(e['phonetic_detail'],e['start'],e['stop']):
                audio = e['audio']['array'][start:stop]
                self.phone.append(
                    {'audio':F.pad(audio[:self.sample_length],(0,max(0,self.sample_length-audio.shape[0])), "constant", 0),
                     'label':label
                    }
                )
        
        
        
    def to_index(self,example):
        dicti = {"phonetic_detail":[]}
        
        dicti = {'start':example['start'],'stop':example['stop'],'phonetic_detail': [self.mapping_phone[x] if x in self.mapping_phone else 38 for x in example['utterance']]}
        return dicti
    
    def __getitem__(self,idx):
        sample = self.phone.__getitem__(idx)
        label = sample['label']
        audio = sample['audio']
        
        #audio = F.pad(audio[:self.sample_length],(0,max(0,self.sample_length-audio.shape[0])), "constant", 0)
        return audio,label
    def __len__(self):
        return len(self.phone)
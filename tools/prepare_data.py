"""builds the readable teaching data and its token cache"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
import hashlib
import json
import re
import shutil
from collections import Counter
import torch
from tokenizers import Tokenizer

# each row has an answer then two practice phrasings then validation then test
FACTS = '''A cat meows.|What sound does a cat make?|What noise does a cat make?|How does a cat sound?|Can you tell me the sound a cat makes?
A dog barks.|What sound does a dog make?|What noise does a dog make?|How does a dog sound?|Can you tell me the sound a dog makes?
A cow moos.|What sound does a cow make?|What noise does a cow make?|How does a cow sound?|Can you tell me the sound a cow makes?
A duck quacks.|What sound does a duck make?|What noise does a duck make?|How does a duck sound?|Can you tell me the sound a duck makes?
A cat.|Which animal says meow?|What animal meows?|Name an animal that meows.|If I hear meowing what animal might it be?
A dog.|Which animal says woof?|What animal barks?|Name an animal that barks.|If I hear barking what animal might it be?
A cow.|Which animal says moo?|What animal moos?|Name an animal that moos.|If I hear mooing what animal might it be?
A duck.|Which animal says quack?|What animal quacks?|Name an animal that quacks.|If I hear quacking what animal might it be?
An umbrella helps keep you dry.|Why do we use umbrellas in rain?|What is an umbrella for?|How does an umbrella help in the rain?|Why would I take an umbrella on a rainy day?
A coat helps keep you warm.|Why do we wear coats in cold weather?|What is a warm coat for?|How does a coat help when it is cold?|Why should I bring a coat on a cold day?
A towel helps you dry off.|Why do we use towels after washing?|What is a towel for?|How does a towel help when you are wet?|What can a towel do after I wash my hands?
Soap helps remove dirt from your hands.|Why do we wash our hands with soap?|What is soap for?|How does soap help clean hands?|Why should I use soap when my hands are dirty?
Plants need water and light to grow.|What do plants need to grow?|What helps a plant grow?|What should I give a plant so it can grow?|What does my little plant need to stay alive?
Drink some water.|What should I do if I am thirsty?|I feel thirsty. What can I drink?|I am thirsty. Any ideas?|My mouth feels dry and I am thirsty. What should I do?
Take a break and get some rest.|What should I do if I am tired?|I feel tired. What can I do?|I am tired after working. Any advice?|I have been studying and feel exhausted. What might help?
You could read a book or draw a picture.|What can I do when I am bored?|I feel bored. Suggest an activity.|Give me something simple to do at home.|I have nothing to do this afternoon. Any ideas?
You could take a short walk.|Suggest something to do outside.|What is a simple outdoor activity?|What could I do outdoors today?|I want to get outside for a little while. Any suggestions?
Wear a warm coat.|What should I wear when it is cold?|Should I wear a coat in cold weather?|It is chilly outside. What should I put on?|Would a coat or a swimsuit be better for cold weather?
The sun gives us light and warmth.|What does the sun give us?|How does the sun help us?|What do we get from the sun?|Why is the sun useful to us?
Ice is frozen water.|What is ice?|What is frozen water called?|Tell me what ice is made of.|What happens to water when it freezes?
A week has seven days.|How many days are in a week?|How long is a week in days?|Tell me the number of days in one week.|If a whole week passes how many days pass?
A triangle has three sides.|How many sides does a triangle have?|What is the number of sides on a triangle?|Count the sides of a triangle.|How many edges go around a triangle?
A square has four sides.|How many sides does a square have?|What is the number of sides on a square?|Count the sides of a square.|How many edges go around a square?
A bicycle has two wheels.|How many wheels does a bicycle have?|What is the number of wheels on a bicycle?|Count the wheels on a bicycle.|How many wheels would I see on a normal bicycle?
The sky is usually blue on a clear day.|What color is the sky on a clear day?|What color does a clear daytime sky look?|Describe the color of a clear sky during the day.|On a sunny day with no clouds what color is the sky?
Grass is usually green.|What color is grass usually?|What is the usual color of grass?|Tell me the color of healthy grass.|If I draw healthy grass what color should I use?
A banana is usually yellow when ripe.|What color is a ripe banana usually?|What is the usual color of a ripe banana?|Tell me the color of a ripe banana.|If I draw a ripe banana what color should I use?
No. Cats meow and dogs bark.|Do cats bark?|Does a cat normally bark?|Is barking the usual sound of a cat?|Would it be right to say that cats bark like dogs?
No. Dogs bark and cats meow.|Do dogs meow?|Does a dog normally meow?|Is meowing the usual sound of a dog?|Would it be right to say that dogs meow like cats?
I cannot know what you have not told me.|What number am I thinking of?|Can you read my mind?|Do you know my secret thought?|Can you tell what I am thinking without me saying it?'''
SOCIAL = [
 ('greeting','Hello! How are you?', ['Hi!','Hello!','Hey!','Good morning!','Hello there!'], ['Hi there, how is it going?'], ['Hey, can we talk for a minute?']),
 ('wellbeing','I am doing well. How are you?', ['How are you?','How are you doing?','Are you doing well?','How is your day?'], ['How are things with you?'], ['Are you having a good day so far?']),
 ('thanks','You are welcome!', ['Thank you!','Thanks!','Thanks for helping me.','I appreciate your help.'], ['Thank you for the help.'], ['That was helpful, thank you.']),
 ('bye','Goodbye! Have a good day.', ['Bye!','Goodbye!','See you later!','I have to go.'], ['I am leaving now.'], ['Time for me to go. See you!']),
 ('sad','I am sorry you feel sad. Would you like to talk about it?', ['I feel sad.','I am feeling down.','I had a bad day.','I am unhappy today.'], ['Today has been a rough day.'], ['I am not feeling happy today.']),
 ('happy','That is good to hear! What made you happy?', ['I feel happy.','I had a great day.','I am excited today.','I am feeling good.'], ['Something nice happened today.'], ['Today has been a really good day for me.']),
 ('learning','Try one small step at a time and take breaks.', ['Learning is hard.','I am struggling to learn.','How can I keep studying?','I feel stuck with my homework.'], ['I am finding my lessons difficult.'], ['I keep getting stuck when I try to learn something.']),
 ('identity','I am a small language model trained to practice basic conversation.', ['Who are you?','What are you?','Tell me about yourself.','Introduce yourself.'], ['Who am I talking to?'], ['Would you introduce yourself to me?']),
 ('question','What did you enjoy doing today?', ['Ask me a question.','Ask about my day.','Can you ask me something?','Give me a question about today.'], ['What would you like to ask me?'], ['Ask me something about how my day went.'])
]
OPPOSITES = [('hot','cold'),('big','small'),('up','down'),('fast','slow'),('happy','sad'),('open','closed'),('wet','dry'),('light','dark'),('near','far'),('full','empty'),('old','new'),('long','short')]


def normalized(text):
    return ' '.join(re.findall(r'\w+', text.casefold()))


def prepare():
    folder = ROOT / 'data/conversations'
    cache_folder = ROOT / 'artifacts/animals'
    if (cache_folder / 'tokens.pt').exists():
        raise ValueError('existing cache will not be overwritten')
    rows = {s: [] for s in ['train','validation','test']}
    def add(split, category, question, answer, context=None, kind='new wording'):
        messages = list(context or []) + [{'role':'user','content':question},{'role':'assistant','content':answer}]
        rows[split].append({'id':f'{split}_{len(rows[split]):04d}','category':category,'test_kind':kind,'messages':messages})
    for line in FACTS.splitlines():
        answer, q1, q2, val, test = line.split('|')
        for question in [q1,q2,'Please answer this: '+q1,'Can you help me? '+q2]:
            add('train','facts',question,answer)
        add('validation','facts',val,answer)
        add('test','facts',test,answer)
    for category, answer, training, validation, testing in SOCIAL:
        for split, questions in [('train',training),('validation',validation),('test',testing)]:
            for question in questions:
                add(split,category,question,answer)
    for first, second in OPPOSITES:
        for word, opposite in [(first,second),(second,first)]:
            answer=f'The opposite of {word} is {opposite}.'
            for question in [f'What is the opposite of {word}?',f'Tell me the opposite of {word}.',f'Which word is the opposite of {word}?']:
                add('train','opposites',question,answer)
            add('validation','opposites',f'What word means the opposite of {word}?',answer)
            add('test','opposites',f'Can you name the opposite of {word}?',answer)
    # some number pairs are held out entirely rather than only changing their wording
    numbers=['zero','one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve']
    for a in range(7):
        for b in range(7):
            bucket=(a*7+b)%7
            split='test' if bucket==5 else 'validation' if bucket==6 else 'train'
            answer=f'There are {numbers[a+b]} in total.'
            questions=[f'I have {numbers[a]} apples and get {numbers[b]} more. How many are there?',f'What is {a} plus {b}?',f'Add {numbers[a]} and {numbers[b]}.']
            for question in questions:
                add(split,'counting',question,answer,kind='held out number pair')
    # names and preferences in the final test are not used in the practice examples
    names={'train':['Anna','Ben','Clara','David','Emma','Jack','Lily','Max','Olivia','Sam','Tom','Zoe'], 'validation':['Alice','Leo','Maya'], 'test':['Nora','Felix','Ruby']}
    hobbies={'train':['reading','drawing','swimming','walking','singing','dancing','cooking','running'], 'validation':['gardening','painting'], 'test':['cycling','knitting']}
    for split, values in names.items():
        for name in values:
            for intro in [f'My name is {name}.', f'You can call me {name}.',f'I am {name}.']:
                context=[{'role':'user','content':intro},{'role':'assistant','content':f'Hello, {name}!'}]
                add(split,'name_intro',intro,f'Hello, {name}!',kind='held out name' if split!='train' else 'practice')
                for question in ['What is my name?','Do you remember my name?','What did I say my name was?']:
                    add(split,'name_memory',question,f'Your name is {name}.',context,kind='held out name' if split!='train' else 'practice')
    for split, values in hobbies.items():
        for hobby in values:
            for intro in [f'I like {hobby}.',f'My hobby is {hobby}.',f'I enjoy {hobby}.']:
                context=[{'role':'user','content':intro},{'role':'assistant','content':f'What do you enjoy about {hobby}?'}]
                add(split,'hobby_intro',intro,f'What do you enjoy about {hobby}?',kind='held out hobby' if split!='train' else 'practice')
                for question in ['What is my hobby?','What do I like doing?','What activity did I mention?']:
                    add(split,'hobby_memory',question,f'You enjoy {hobby}.',context,kind='held out hobby' if split!='train' else 'practice')
    expand_curriculum(rows, add)
    animal_curriculum(rows, add)
    cache_folder.mkdir(parents=True, exist_ok=True)
    folder.mkdir(parents=True, exist_ok=True)
    tokenizer=Tokenizer.from_file(str(ROOT/'configs/tokenizer.json'))
    ids={n:tokenizer.token_to_id('<|'+n+'|>') for n in ['pad','visitor','radahn','eos']}
    # reuse local english tokens when available
    previous = ROOT/'artifacts/conversation/tokens.pt'
    old = torch.load(previous, weights_only=True, map_location='cpu') if previous.exists() else None
    cache={'streams':{},'dialogues':{}}
    signatures={}
    for split, records in rows.items():
        unique = {}
        for row in records:
            key = normalized(json.dumps(row['messages'][:-1], sort_keys=True))
            if key in unique:
                # these equivalent answers were reviewed when merging the animal examples
                allowed = {'Name an animal that moos.', 'Name an animal that quacks.', 'Is a penguin a bird?'}
                assert unique[key]['messages'][-1] == row['messages'][-1] or row['messages'][-2]['content'] in allowed
            unique[key] = row
        records[:] = list(unique.values())
        for i, row in enumerate(records):
            row['id'] = f'{split}_{i:04d}'
        signatures[split]=set()
        inputs,targets,flat=[],[],[]
        for row in records:
            messages=row['messages']; seq=[]; start=0
            prompt=messages[:-1]
            signature=normalized(json.dumps(prompt,sort_keys=True))
            assert signature not in signatures[split]
            signatures[split].add(signature)
            for i,m in enumerate(messages):
                assert m['role']==('user' if i%2==0 else 'assistant')
                seq.append(ids['visitor' if i%2==0 else 'radahn'])
                if i==len(messages)-1:start=len(seq)
                seq.extend(tokenizer.encode(m['content']).ids)
                if i%2:seq.append(ids['eos'])
            assert len(seq)<=257
            x=torch.full((256,),ids['pad'],dtype=torch.long)
            y=torch.full((256,),-100,dtype=torch.long)
            x[:len(seq)-1]=torch.tensor(seq[:-1])
            y[start-1:len(seq)-1]=torch.tensor(seq[start:])
            assert torch.all(y[:start-1]==-100) and y[len(seq)-2]==ids['eos']
            inputs.append(x);targets.append(y);flat.extend(seq)
        (folder/(split+'.jsonl')).write_text('\n'.join(json.dumps(r) for r in records)+'\n',encoding='utf-8')
        cache['dialogues'][split]=(torch.stack(inputs),torch.stack(targets))
        cache['streams'][split]=torch.tensor(flat,dtype=torch.long)
        if old is not None:
            cache['streams']['english_'+split]=old['streams']['english_'+split]
        elif (ROOT/'data/english'/(split+'.txt')).exists():
            text=(ROOT/'data/english'/(split+'.txt')).read_text(encoding='utf-8')
            cache['streams']['english_'+split]=torch.tensor(tokenizer.encode(text).ids, dtype=torch.long)
    assert not signatures['train'] & (signatures['validation']|signatures['test'])
    assert not signatures['validation'] & signatures['test']
    torch.save(cache,cache_folder/'tokens.pt')
    shutil.copyfile(ROOT/'configs/tokenizer.json',cache_folder/'tokenizer.json')
    manifest={'vocab_size':4096,'context_length':256,'counts':{s:dict(Counter(r['category'] for r in rs)) for s,rs in rows.items()},'total':{s:len(rs) for s,rs in rows.items()},'tokenizer_sha256':hashlib.sha256((cache_folder/'tokenizer.json').read_bytes()).hexdigest(),'cache_sha256':hashlib.sha256((cache_folder/'tokens.pt').read_bytes()).hexdigest(),'source':'locally authored synthetic teaching examples in tools/prepare_data.py','split_policy':'fresh animal questions and held out fictional names and food home combinations with earlier basic training retained','limitations':'narrow template based curriculum does not demonstrate broad language understanding','verification':'no identical full prompts across splits and checked answer masks context size and eos'}
    (cache_folder/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (ROOT/'data/sources/animal-training.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))



def expand_curriculum(rows, add):
    # the old test has been examined so it is practice now rather than a hidden test
    rows['train'] = rows['train'] + rows['validation'] + rows['test']
    rows['validation'] = []
    rows['test'] = []
    variations = {
        'greeting': 'Good evening!|Good afternoon!|Hi there!|Hey there!|Hello, I would like to chat.|Can we talk?|I want to chat with you.|Hello, can we have a conversation?|Hi, nice to meet you.|Hey, are you there?|Hello again.|I just wanted to say hello.',
        'wellbeing': 'How is everything going?|How are you feeling today?|How has your day been?|Are you okay?|How have you been?|How is life treating you?|Is your day going well?|Are things going well for you?|Hi, how are you doing?',
        'thanks': 'Thanks a lot.|That helped me.|Thank you very much.|I am grateful for your help.|Thanks for the answer.|I appreciate that.|You helped me out.|Many thanks.',
        'bye': 'I need to leave now.|Talk to you later.|Have a good day. Goodbye.|See you soon.|I am done chatting for now.|I will come back later.|Bye for now.|Catch you later.',
        'sad': 'I had a rough day.|I am having a difficult day.|Things went badly today.|I am upset.|I feel lonely.|I feel miserable.|I am disappointed today.|I could use someone to talk to.|I feel unhappy.|My day has been awful.',
        'happy': 'I have good news.|I feel great.|Something wonderful happened.|I am in a good mood.|I am really pleased today.|I am happy with how today went.|I had fun today.|I am feeling cheerful.',
        'learning': 'I am stuck on my schoolwork.|This lesson is hard for me.|Studying feels difficult.|How do I learn something hard?|I need help staying motivated to study.|I made a mistake on my homework.|I failed a quiz.|I am struggling with class.|I am having trouble learning.',
        'identity': 'What kind of model are you?|Are you a chatbot?|Explain what you are.|What were you made to do?|Tell me who you are.|Can you introduce yourself?|Who is answering my questions?',
        'question': 'Ask me about today.|Please ask me a question.|I want you to ask about my day.|What question do you have for me?|Give me something to answer.|Can you ask how my day went?|Ask me about what I did today.'
    }
    for category, answer, _, _, _ in SOCIAL:
        for question in variations[category].split('|'):
            for prefix in ['', 'Hi. ', 'Hey. ', 'Hello. ']:
                add('train',category,prefix+question,answer)
    # use more everyday wording for the same basic facts
    for line in FACTS.splitlines():
        answer,*questions = line.split('|')
        for q in questions:
            for prefix in ['I have a question. ', 'Please help me understand. ', 'I was wondering: ']:
                add('train','facts',prefix+q,answer)
    extra_facts = [
        ('An umbrella helps keep you dry.','How can I stay dry in the rain?|What should I carry on a rainy walk?|It is raining. What can keep me dry?|I need to walk in rain. What should I bring?'),
        ('A towel helps you dry off.','My hands are wet. What can dry them?|What should I use to dry my hands?|How can I dry off after a bath?'),
        ('Wear a warm coat.','It is freezing outside. What can I wear?|I need to stay warm outside. What should I wear?|Should I choose a coat or a swimsuit for a cold day?'),
        ('Take a break and get some rest.','I feel exhausted. What could help?|I am worn out after class. What might help?|I studied all day and need a rest.|I am sleepy after a long day.|I am tired from studying.'),
        ('Drink some water.','I need a drink. What should I have?|I am thirsty after a walk.|What can help when I feel thirsty?'),
        ('You could read a book or draw a picture.','I have nothing to do. Any suggestions?|I want a quiet activity at home.|What can I do for fun indoors?|I am bored at home. Help me pick an activity.'),
        ('A cat meows.','Which pet usually makes a meowing noise?|What noise would I hear from a cat?'),
        ('A dog barks.','What noise would I hear from a dog?|What sound might a dog make?'),
    ]
    for answer,questions in extra_facts:
        for q in questions.split('|'):
            for prefix in ['', 'Please help. ', 'Hello. ']:add('train','facts',prefix+q,answer)
    words='zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty'.split()
    # practice all small sums then test new wording rather than claiming new arithmetic
    for a in range(11):
        for b in range(11):
            answer=f'There are {words[a+b]} in total.'
            for q in [f'What is {a} plus {b}?',f'Add {words[a]} and {words[b]}.',f'I have {words[a]} apples and get {words[b]} more. How many are there?',f'I have {words[a]} coins. I find {words[b]} more. How many coins do I have?',f'There are {a} birds and {b} more arrive. How many birds are there now?']:
                add('train','counting',q,answer,kind='practiced small sums')
    names='Anna Ben Clara David Emma Jack Lily Max Olivia Sam Tom Zoe Alice Leo Maya Nora Felix Ruby Elena Adam Amy Daniel Ella Grace Henry Isaac Julia Liam Lucy Maria Noah Oliver Oscar Peter Rose Sarah Theo Violet William'.split()
    hobbies=['reading','drawing','swimming','walking','singing','dancing','cooking','running','gardening','painting','cycling','knitting','baking','hiking','fishing','camping','chess','football','tennis','basketball','photography','writing','puzzles','skating','skiing','sewing','yoga','music','movies','video games','making models','playing the piano','playing the guitar','collecting stamps','watching birds','building things']
    for category,values in [('name',names),('hobby',hobbies)]:
        for value in values:
            if category=='name':
                introductions=[f'People call me {value}.',f'Hi, my name is {value}.',f'My friends call me {value}.']
                reply=f'Hello, {value}!';answer=f'Your name is {value}.'
                questions=['What is my name again?','Tell me the name I gave you.','Can you remember what I am called?']
            else:
                introductions=[f'I am into {value}.',f'For fun, I enjoy {value}.',f'I spend my free time {value}.']
                reply=f'What do you enjoy about {value}?';answer=f'You enjoy {value}.'
                questions=['What activity did I tell you about?','What do I enjoy doing for fun?','Remind me what my hobby is.']
            for intro in introductions:
                add('train',category+'_intro',intro,reply)
                context=[{'role':'user','content':intro},{'role':'assistant','content':reply}]
                for q in questions:
                    add('train',category+'_memory',q,answer,context)
                # a short unrelated exchange should not erase the earlier detail
                context=context+[{'role':'user','content':'Thanks.'},{'role':'assistant','content':'You are welcome!'}]
                add('train',category+'_memory',questions[0],answer,context)
    validation = [
        ('greeting','Hello, is anyone here to chat with me?','Hello! How are you?'),
        ('wellbeing','Has today been treating you well?','I am doing well. How are you?'),
        ('thanks','That answer was useful. I appreciate it.','You are welcome!'),
        ('bye','I must head out. Until next time!','Goodbye! Have a good day.'),
        ('sad','I am feeling pretty low this evening.','I am sorry you feel sad. Would you like to talk about it?'),
        ('happy','I am delighted with my day.','That is good to hear! What made you happy?'),
        ('learning','My lessons are confusing and I want to quit.','Try one small step at a time and take breaks.'),
        ('identity','Tell me a little about the model I am chatting with.','I am a small language model trained to practice basic conversation.'),
        ('question','Could you ask about something I enjoyed today?','What did you enjoy doing today?'),
        ('facts','What noise should I expect a duck to make?','A duck quacks.'),
        ('facts','I washed my hands. What can help dry them?','A towel helps you dry off.'),
        ('facts','Rain is coming and I want to stay dry. What helps?','An umbrella helps keep you dry.'),
        ('facts','I have very little energy after studying. What could help?','Take a break and get some rest.'),
        ('facts','What keeps a little plant growing?','Plants need water and light to grow.'),
        ('facts','What color would you use for healthy grass in a drawing?','Grass is usually green.'),
        ('facts','How many sides would I count on a square?','A square has four sides.'),
        ('facts','Would a dog usually make the same noise as a cat?','No. Dogs bark and cats meow.'),
        ('facts','What sort of light and heat do we get from the sun?','The sun gives us light and warmth.'),
    ]
    testing = [
        ('greeting','Good evening. Is this a good time to talk?','Hello! How are you?'),
        ('wellbeing','Are you feeling all right this evening?','I am doing well. How are you?'),
        ('thanks','Your help meant a lot. Thank you.','You are welcome!'),
        ('bye','I should get going now. Take care!','Goodbye! Have a good day.'),
        ('sad','Nothing went well for me and I feel down.','I am sorry you feel sad. Would you like to talk about it?'),
        ('happy','I have had a wonderful afternoon and feel cheerful.','That is good to hear! What made you happy?'),
        ('learning','I cannot get the hang of my homework yet.','Try one small step at a time and take breaks.'),
        ('identity','How would you describe yourself as a chatbot?','I am a small language model trained to practice basic conversation.'),
        ('question','Would you ask me what was nice about today?','What did you enjoy doing today?'),
        ('facts','What sort of noise comes from a cow?','A cow moos.'),
        ('facts','After a shower I am wet. What can dry me?','A towel helps you dry off.'),
        ('facts','I need to go outdoors in a downpour. What will keep me dry?','An umbrella helps keep you dry.'),
        ('facts','I feel worn out from my schoolwork. What should I try?','Take a break and get some rest.'),
        ('facts','My plant needs to grow. What should it get?','Plants need water and light to grow.'),
        ('facts','What color would a ripe banana be in a picture?','A banana is usually yellow when ripe.'),
        ('facts','If I count around a triangle how many sides will I find?','A triangle has three sides.'),
        ('facts','Is it normal for a cat to sound like a barking dog?','No. Cats meow and dogs bark.'),
        ('facts','Can you explain what water becomes when it is frozen?','Ice is frozen water.'),
    ]
    for split,examples in [('validation',validation),('test',testing)]:
        for category,q,a in examples:add(split,category,q,a,kind='fresh authored wording')
        for word,opposite in [('hot','cold'),('big','small'),('up','down'),('fast','slow')]:
            q=f'Give me a word meaning the reverse of {word}.' if split=='validation' else f'If something is not {word} what is its opposite called?'
            add(split,'opposites',q,f'The opposite of {word} is {opposite}.',kind='fresh authored wording')
        for a,b in [(2,3),(4,2),(3,5),(6,1)]:
            q=f'I start with {words[a]} coins and receive {words[b]} extra coins. How many now?' if split=='validation' else f'{words[a].capitalize()} apples are in my bag. I put {words[b]} extra apples in it. How many altogether?'
            add(split,'counting',q,f'There are {words[a+b]} in total.',kind='new wording of practiced sums')
        for category,values in [('name',['Jonah','Elsie'] if split=='validation' else ['Amira','Xavier']),('hobby',['origami','woodworking'] if split=='validation' else ['pottery','juggling'])]:
            for value in values:
                intro=f'People call me {value}.' if category=='name' else f'For fun, I enjoy {value}.'
                reply=f'Hello, {value}!' if category=='name' else f'What do you enjoy about {value}?'
                answer=f'Your name is {value}.' if category=='name' else f'You enjoy {value}.'
                q='What is my name again?' if category=='name' else 'What activity did I tell you about?'
                context=[{'role':'user','content':intro},{'role':'assistant','content':reply}]
                add(split,category+'_intro',intro,reply,kind='unseen value')
                add(split,category+'_memory',q,answer,context,kind='unseen value')
                context=context+[{'role':'user','content':'Thanks.'},{'role':'assistant','content':'You are welcome!'}]
                add(split,category+'_memory',q,answer,context,kind='unseen value after intervening exchange')



def animal_curriculum(rows, add):
    import random
    # keep some earlier practice so learning animals does not replace every old skill
    previous = rows['train']
    rng = random.Random(448)
    rows['train'] = []
    for category in sorted({r['category'] for r in previous}):
        candidates = [r for r in previous if r['category'] == category]
        rows['train'].extend(rng.sample(candidates, min(100, len(candidates))))
    rows['validation'] = []
    rows['test'] = []
    animals = json.loads((ROOT/'data/sources/animal_facts.json').read_text())['animals']
    def article(word):
        return 'an' if word[0].lower() in 'aeiou' else 'a'
    for animal in animals:
        name,group=animal['name'],animal['group']
        subject=f'{article(name).capitalize()} {name}'
        classification=f'{subject} is {article(group)} {group}.'
        description=classification+' '+animal['feature']
        questions=[f'What kind of animal is {article(name)} {name}?',f'Which animal group does {article(name)} {name} belong to?',f'Is {article(name)} {name} {article(group)} {group}?',f'Classify {article(name)} {name}.']
        for q in questions:
            for prefix in ['', 'Please explain. ', 'I am learning about animals. ']:
                add('train','animal_groups',prefix+q,classification)
        for q in [f'Tell me about {article(name)} {name}.',f'What is {article(name)} {name}?',f'Describe {article(name)} {name}.',f'Give me a fact about {article(name)} {name}.']:
            add('train','animal_descriptions',q,description)
        for q in [f'Where does {article(name)} {name} live?',f'What is the habitat of {article(name)} {name}?',f'Tell me where {article(name)} {name} can live.']:
            add('train','animal_habitats',q,animal['habitat'])
        for q in [f'What is special about {article(name)} {name}?',f'Tell me a feature of {article(name)} {name}.',f'What should I know about {article(name)} {name}?']:
            add('train','animal_features',q,animal['feature'])
        if animal['sound']:
            sound=animal['sound'];answer=f'{subject} {sound}.'
            for q in [f'What sound does {article(name)} {name} make?',f'What noise does {article(name)} {name} make?',f'How does {article(name)} {name} sound?',f'Which animal {sound}?',f'What animal {sound}?',f'Name an animal that {sound}.']:
                # the bee and mosquito share a sound so a reverse question needs a unique answer
                if sound=='buzzes' and q.startswith(('Which','What animal','Name')):
                    continue
                add('train','animal_sounds',q,answer)
        intro=f'Let us talk about the {name}.'
        context=[{'role':'user','content':intro},{'role':'assistant','content':description}]
        add('train','animal_followup',intro,description)
        for q,answer in [('What group does it belong to?',classification),('Where does it live?',animal['habitat']),('What is special about it?',animal['feature'])]:
            add('train','animal_followup',q,answer,context)
        for other in animals:
            if other['name']==name or other['group']==group:
                continue
            # contrast groups instead of implying that every flying animal is a bird
            if other['name'] not in ['dog','duck','frog','shark']:
                continue
            answer=f"The {name} is {article(group)} {group}, while the {other['name']} is {article(other['group'])} {other['group']}."
            for q in [f'Are the {name} and the {other["name"]} in the same animal group?',f'Compare the animal groups of a {name} and a {other["name"]}.']:
                add('train','animal_comparison',q,answer)
    lessons = [
        ('Dogs commonly bark.', 'What animal makes a barking sound?|Which pet makes a barking sound?|What animal goes woof?|Which animal do you hear barking?|I heard a bark. What animal could it be?'),
        ('Birds have feathers and lay eggs. Many birds can fly, but some cannot.', 'What are birds?|Tell me what a bird is.|What makes an animal a bird?|Describe birds.|Explain birds to me.'),
        ('No. Penguins and ostriches are birds that cannot fly.', 'Can all birds fly?|Does every bird fly?|Are all birds able to fly?|Do birds always fly?|Must a bird be able to fly?'),
        ('No. A bat is a mammal, not a bird.', 'Is a bat a bird?|Are bats birds?|Does having wings make a bat a bird?|Is a flying bat a kind of bird?'),
        ('A penguin is a bird even though it cannot fly.', 'Is a penguin a bird?|Are penguins birds?|Can a penguin be a bird if it cannot fly?|Does being unable to fly mean a penguin is not a bird?'),
        ('A whale is a mammal. It breathes air and feeds its young milk.', 'Is a whale a fish or a mammal?|Are whales fish?|Why is a whale a mammal?|Does living in water make a whale a fish?'),
        ('A dolphin is a mammal and breathes air with lungs.', 'Is a dolphin a fish?|What makes a dolphin different from a fish?|Do dolphins breathe with gills?|How does a dolphin breathe?'),
        ('A spider is an arachnid with eight legs. An adult insect has six legs.', 'Is a spider an insect?|Are spiders insects?|How is a spider different from an insect?|Does a spider have six legs?'),
        ('By human deaths from transmitted diseases, mosquitoes are considered the deadliest animals.', 'What is the most dangerous animal?|Which animal is deadliest to humans?|What animal causes the most human deaths through disease?|What is the deadliest animal?|Which animal is most dangerous to people overall?'),
        ('Some mosquitoes transmit germs that cause diseases such as malaria. That makes them dangerous to people.', 'Why are mosquitoes dangerous?|How can a tiny mosquito be deadly?|Why can mosquitoes harm people?|What makes mosquitoes so dangerous?'),
        ('It depends on the measure. Mosquitoes spread deadly diseases, while large predators can cause direct injuries.', 'Is a mosquito more dangerous than a lion?|Which is worse a lion or a mosquito?|Does most dangerous mean the biggest animal?|Does the strongest animal always cause the most deaths?'),
        ('Not every mosquito carries disease, and not every bite makes someone sick.', 'Does every mosquito bite cause disease?|Are all mosquitoes infected?|Does every mosquito carry malaria?'),
        ('Leave wild animals alone and keep a safe distance.', 'Should I touch a wild snake?|Can I approach a wild lion?|Should I pick up an animal I do not know?'),
    ]
    for answer,qs in lessons:
        for q in qs.split('|'):
            for prefix in ['', 'Please answer. ', 'I am curious about animals. ', 'Can you explain this? ']:
                add('train','animal_reasoning',prefix+q,answer)
    # invented story animals let facts change without pretending that real biology changes
    foods=['seeds','berries','fish','leaves','fruit','insects']
    homes=['a cave','a forest','a pond','a grassland']
    names={'train':['mip','nalo','tavi','luma','beko','suri','fep','toma'],'validation':['wug','dax'],'test':['niff','zorp']}
    held=[(i,j) for i in range(len(foods)) for j in range(len(homes)) if (i+j)%5==0]
    for split,values in names.items():
        pairs=[(i,j) for i in range(len(foods)) for j in range(len(homes)) if (i,j) not in held] if split=='train' else held[:2] if split=='validation' else held[2:]
        for name in values:
            for i,j in pairs:
                food,home=foods[i],homes[j]
                if split=='train':
                    passages=[f'In a made-up story, a {name} eats {food} and lives in {home}.',f'In this fictional story, a {name} lives in {home}. Its food is {food}.']
                elif split=='validation':
                    passages=[f'Read this invented description: the {name} lives in {home} and eats {food}.']
                else:
                    passages=[f'Here is a fictional creature called a {name}. It eats {food}. Its home is {home}.']
                for passage in passages:
                    questions=[('What does it eat?',f'It eats {food}.'),('Where does it live?',f'It lives in {home}.')]
                    for q,answer in questions:
                        add(split,'read_description',passage+' '+q,answer,kind='unseen fictional name and food home pairing' if split!='train' else 'practice using supplied information')
    # these phrasings are reserved and are never included in the practice examples
    fresh = {
        'validation': [
            ('animal_sounds','Which animal might I hear making a woof sound?','A dog barks.'),
            ('animal_sounds','If I hear a meow which pet is likely nearby?','A cat meows.'),
            ('animal_sounds','What is the usual noise made by a cow?','A cow moos.'),
            ('animal_sounds','What sort of call might come from a duck?','A duck quacks.'),
            ('animal_groups','Would you put a frog in the amphibian group?','A frog is an amphibian.'),
            ('animal_groups','Which group of animals includes butterflies?','A butterfly is an insect.'),
            ('animal_groups','What group should an elephant be placed in?','An elephant is a mammal.'),
            ('animal_groups','Would a shark be classified as a fish?','A shark is a fish.'),
            ('animal_reasoning','Can you explain why bats are not birds?','No. A bat is a mammal, not a bird.'),
            ('animal_reasoning','If penguins cannot fly are they still birds?','A penguin is a bird even though it cannot fly.'),
            ('animal_reasoning','Is flight something every bird can do?','No. Penguins and ostriches are birds that cannot fly.'),
            ('animal_reasoning','What features make whales mammals instead of fish?','A whale is a mammal. It breathes air and feeds its young milk.'),
            ('animal_danger','Which animal is called the deadliest because of diseases it spreads?','By human deaths from transmitted diseases, mosquitoes are considered the deadliest animals.'),
            ('animal_danger','How do mosquitoes cause harm despite their small size?','Some mosquitoes transmit germs that cause diseases such as malaria. That makes them dangerous to people.'),
            ('animal_danger','Is every mosquito carrying a disease?','Not every mosquito carries disease, and not every bite makes someone sick.'),
            ('animal_features','What body part helps an elephant grab things?','Elephants use their trunks to grasp things.'),
            ('animal_features','How many legs does a spider have compared with an insect?','A spider is an arachnid with eight legs. An adult insect has six legs.'),
            ('social','Hello there, could we discuss animals?','Hello! How are you?'),
        ],
        'test': [
            ('animal_sounds','Name a pet whose sound is barking.','A dog barks.'),
            ('animal_sounds','Which farm animal commonly makes a mooing noise?','A cow moos.'),
            ('animal_sounds','What sound would I associate with a sheep?','A sheep bleats.'),
            ('animal_sounds','What noise is a horse known for making?','A horse neighs.'),
            ('animal_groups','Does an owl belong with the birds or the mammals?','An owl is a bird.'),
            ('animal_groups','Which category of animal would a crocodile fit into?','A crocodile is a reptile.'),
            ('animal_groups','Should I group a bee with the insects?','A bee is an insect.'),
            ('animal_groups','What kind of animal group contains rabbits?','A rabbit is a mammal.'),
            ('animal_reasoning','A bat has wings so should I call it a bird?','No. A bat is a mammal, not a bird.'),
            ('animal_reasoning','A penguin cannot take off into the sky. Does that stop it being a bird?','A penguin is a bird even though it cannot fly.'),
            ('animal_reasoning','Can you give examples showing that birds do not all fly?','No. Penguins and ostriches are birds that cannot fly.'),
            ('animal_reasoning','Whales swim in the ocean. Why are they not classed as fish?','A whale is a mammal. It breathes air and feeds its young milk.'),
            ('animal_danger','What animal is most deadly when we count human deaths from disease?','By human deaths from transmitted diseases, mosquitoes are considered the deadliest animals.'),
            ('animal_danger','How can something as small as a mosquito endanger people?','Some mosquitoes transmit germs that cause diseases such as malaria. That makes them dangerous to people.'),
            ('animal_danger','Is becoming ill guaranteed whenever a mosquito bites?','Not every mosquito carries disease, and not every bite makes someone sick.'),
            ('animal_features','What helps a giraffe reach leaves high up?','Giraffes have long necks that help them reach leaves.'),
            ('animal_features','Why would it be wrong to call an eight legged spider an insect?','A spider is an arachnid with eight legs. An adult insect has six legs.'),
            ('social','Hi, I would love to chat about animals.','Hello! How are you?'),
        ]
    }
    for split,examples in fresh.items():
        for category,q,answer in examples:
            add(split,category,q,answer,kind='held out animal phrasing')
        for name in ['penguin','bat','whale']:
            animal=next(a for a in animals if a['name']==name)
            classification=f'{article(name).capitalize()} {name} is {article(animal["group"])} {animal["group"]}.'
            intro=f'Can we discuss the {name} now?' if split=='validation' else f'I want to learn about the {name}.'
            context=[{'role':'user','content':intro},{'role':'assistant','content':classification+' '+animal['feature']}]
            question='What group is it part of?' if split=='validation' else 'What kind of animal is that?'
            add(split,'animal_followup',question,classification,context,kind='held out conversational phrasing')


if __name__ == '__main__':
    prepare()

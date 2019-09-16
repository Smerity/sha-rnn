import argparse

import torch
import torch.nn.functional as F

import data

parser = argparse.ArgumentParser(description='PyTorch PTB Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='output.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
args = parser.parse_args()

def model_save(fn):
    with open(fn, 'wb') as f:
        #torch.save([model, criterion, optimizer], f)
        torch.save([model, criterion], f)

def model_load(fn):
    global model, criterion, optimizer
    with open(fn, 'rb') as f:
        #model, criterion, optimizer = torch.load(f)
        m, criterion = torch.load(f)
        model.load_state_dict(m.state_dict(), strict=False)
        del m

model, criterion = torch.load(args.checkpoint)

model.eval()

if args.cuda:
    model.cuda()
else:
    model.cpu()

import os
import hashlib
fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
if os.path.exists(fn):
    print('Loading cached dataset...')
    corpus = torch.load(fn)
else:
    print('Producing dataset...')
    corpus = data.Corpus(args.data)
    torch.save(corpus, fn)

dictionary = corpus.dictionary
del corpus
ntokens = len(dictionary)
hidden = None
mems = None

text = b'''Specialists in Japanese [[historical linguistics]] all agree that Japanese is related to the [[Ryukyuan languages]] (including [[Okinawan language|Okinawan]]); together, Japanese and Ryukyuan are grouped in the [[Japonic languages]]. Among these specialists, the possibility of a genetic relation to [[Goguryeo]] has the most evidence; relationship to [[Korean language|Korean]] is considered plausible but is still up to debate; the Altaic hypothesis has somewhat less currency, though it has grown significantly more respectable in recent years, primarily due to the work of [[Sergei Starostin]], et al. Almost all specialists reject the idea that Japanese could'''
text = b'''Neopets allows users to create and care for digital pets called "Neopets" and explore the virtual world of Neopia. There is no set objective for the users, but they are expected to feed and care for their Neopets when they grow hungry or ill. Neopets will not die if neglected, but their health can limit their gameplay.[2] Neopets come in a variety of species and colors and users can create or adopt their own. Users can obtain items to interact with their Neopet, such as books to read and toys to play with them. Neopets can be customised with certain clothing items, paint brushes, transformation potions, and accessories. Users can build a customisable Neohome for their Neopets and furnish it with furniture, wallpaper, and flooring. Neopets can battle against other Neopets or non-player characters in the Battledome but they cannot die there.


A customisable Neohome.
Neopia is a virtual planet with fantasy lands inhabited by Neopets and other virtual creatures. Each land has a different theme, such as pirates or prehistory, and their own shops, games, and attractions.[3] Neopia follows its own calendar and time zone, which runs concurrent with real-world Pacific Time,[4] and has tie-ins with certain real-world holidays such as Halloween and Christmas. It has its own economy and stock market based on Neopoints. Users can earn Neopoints through various means including playing games and selling items, which can be invested or used to buy various virtual goods and services.[5] While there is no set objective for users, interactive storylines are sometimes released that introduce changes to the planet such as new lands.

The site is regularly updated with features like new games, items and content. In addition to the site content updated by the Neopets staff members, users also contribute content to the site.[6] User contributions come in the form of prescreened submissions and readily editable content that is automatically filtered, such as the site's weekly electronic newspaper The Neopian Times. There are different types of submissions that will be accepted.'''
text = b'''Google was founded in 1998 by [[Larry Page]] and [[Sergey Brin]] while they were Ph.D. students at [[Stanford University]] in [[California]]. Together they own about 14 percent of its shares and control 56 percent of the stockholder voting power through [[Preferred stock|supervoting]] stock. They incorporated Google as a California privately held company on September 4, 1998 in California. Google was then reincorporated in Delaware on October 22, 2002.<ref>{{Cite web|url=https://businesssearch.sos.ca.gov/Document/RetrievePDF?Id=02474131-5043839|title=Business Entity Filing|last=|first=|date=October 7, 2002|website=Business Search|archive-url=https://web.archive.org/web/20190814022055/https://businesssearch.sos.ca.gov/Document/RetrievePDF?Id=02474131-5043839|archive-date=August 14, 2019|dead-url=no|access-date=August 14, 2019}}</ref> An initial public offering (IPO) took place on August 19, 2004, and Google moved to its headquarters in Mountain View, California, nicknamed the [[Googleplex]]. In August 2015, Google announced plans to reorganize its various interests as a conglomerate called [[Alphabet Inc.]] Google is Alphabet's leading subsidiary and will continue to be the umbrella company for Alphabet's Internet interests. [[Sundar Pichai]] was appointed CEO of Google, replacing Larry Page who became the CEO of Alphabet.'''
#text = b"""[[Image:Declaration of Human Rights.jpg|thumb|right|The principles from the French [[Declaration of the Rights of Man and of the Citizen]] still have constitutional importance]]
#'''Constitutional law''' is a body of law which defines the role, powers, and structure of different entities within a [[State (polity)|state]], namely, the [[executive (government)|executive]], the [[parliament]] or [[legislature]], and the [[judiciary]]; as well as the basic rights of citizens and, in federal countries [[History of the United States Constitution|such as the United States]] and [[Provinces of Canada|Canada]], the relationship between the central government and state, provincial, or territorial governments.
#
#Not all [[nation state]]s have codified [[constitution]]s, though all such states have a ''[[jus commune]]'', or law of the land, that may consist of a variety of imperative and consensual rules. These may include [[custom (law)|customary law]], [[Convention (norm)|conventions]], [[statutory law]], [[precedent|judge-made law]], or [[international law|international rules and norms]]. Constitutional law deals with the fundamental principles by which the government exercises its authority. In some instances, these principles grant specific powers to the government, such as the power to tax and spend for the welfare of the population. Other times, constitutional principles act to place limits on what the government can do, such as prohibiting the arrest of an individual without sufficient cause.
#
#==State and legal structure==
#
#Constitutional laws can be"""
#text = b"""The '''Supreme Court of the United States''' ('''SCOTUS''')<ref>{{cite news|last1=Safire|first1=William|url=https://www.nytimes.com/1997/10/12/magazine/on-language-potus-and-flotus.html|title=On language: POTUS and FLOTUS|newspaper=[[The New York Times Magazine]]|date=October 12, 1997|accessdate=August 27, 2013 | authorlink=William Safire}}</ref> is the [[Supreme court|highest court]] in the [[Federal judiciary of the United States|federal judiciary]] of the [[United States|United States of America]], established pursuant to [[Article Three of the United States Constitution|Article III]] of the [[Constitution of the United States|U.S. Constitution]] in 1789. It has ultimate (and largely [[Procedures of the Supreme Court of the United States|discretionary]]) [[appellate jurisdiction]] over all federal and [[State court (United States)|state court]] cases that involve a point of [[Law of the United States|federal law]], and [[original jurisdiction]] over a narrow range of cases, including suits between two or more [[U.S. state|states]] and those involving [[ambassador]]s. The Court holds the power of [[judicial review]], the ability to invalidate a statute for violating a provision of the Constitution. [[Presidential directive]]s can be struck down by the Court for violating either the Constitution or statutory law.<ref name=aboutSC>{{cite web| title=About the Supreme Court| url=http://www.uscourts.gov/about-federal-courts/educational-resources/about-educational-outreach/activity-resources/about| publisher=[[Administrative Office of the United States Courts]]| location=Washington, D.C.| accessdate=September 3, 2018}}</ref> However, it may act only within the context of a case in an area of law over which it has jurisdiction. The Court may decide cases having political overtones, but it has ruled that it does not have power to decide [[Justiciability|non-justiciable]] [[political question]]s.
#
#As set by the [[Judiciary Act of 1869]], the Court consists of the [[Chief Justice of the United States|chief justice of the United States]] and eight [[Associate Justice of the Supreme Court of the United States|associate justices]]. Each justice has [[life tenure|lifetime tenure]], meaning they remain on the Court until they resign, retire, die, or are [[Impeachment in the United States|removed from office]]."""
#text = b"""A '''compiler''' is a [[computer program]] that [[Translator (computing)|translates]] computer code written in one [[programming language]] (the source language) into another language (the target language). The name ''compiler'' is primarily used for programs that translate [[source code]] from a [[high-level programming language]] to a [[lower level language]] (e.g., [[assembly language]], [[object code]], or [[machine code]]) to create an [[executable]] program.<ref>{{cite web| author = PC Mag Staff | date = 28 February 2017 | title = Encyclopedia: Definition of Compiler | work = PCMag.com | url=https://www.pcmag.com/encyclopedia/term/40105 | accessdate=28 February 2017}}</ref><ref name=dragon>[[Compilers: Principles, Techniques, and Tools]] by Alfred V. Aho, Ravi Sethi, Jeffrey D. Ullman - Second  Edition, 2007</ref>{{rp|p1}}
#
#However, there are many different types of compilers. If the compiled program can run on a computer whose [[Central processing unit|CPU]] or [[operating system]] is different from the one on which the compiler runs, the compiler is a [[cross-compiler]]. A [[bootstrap compiler]] is written in the language that it intends to compile. A program that translates from a [[low-level language]] to a higher level one is a [[decompiler]]. A program that translates between high-level languages is usually called a [[source-to-source compiler]] or transpiler. A language [[rewriting|rewriter]] is usually a program that translates the form of expressions without a change of language. The term [[compiler-compiler]] refers to tools used to create parsers that perform syntax analysis."""
#text = """A '''neural network''' is a network or circuit of [[neuron]]s, or in a modern sense, an [[artificial neural network]], composed of [[artificial neuron]]s or nodes.<ref>{{cite journal |first=J. J. |last=Hopfield |title=Neural networks and physical systems with emergent collective computational abilities |journal=Proc. Natl. Acad. Sci. U.S.A. |volume=79 |issue= 8|pages=2554–2558 |year=1982 |doi=10.1073/pnas.79.8.2554 |pmc=346238 }}</ref> Thus a neural network is either a [[biological neural network]], made up of real biological neurons, or an artificial neural network, for solving [[artificial intelligence]] (AI) problems. The connections of the biological neuron are modeled as weights. A positive weight reflects an excitatory connection, while negative values mean inhibitory connections. All inputs are modified by a weight and summed. This activity is referred as a linear combination. Finally, an activation function controls the [[amplitude]] of the output. For example, an acceptable range of output is usually between 0 and 1, or it could be −1 and 1.
#
#These artificial networks may be used for [[predictive modeling]], adaptive control and applications where they can be trained via a dataset. Self-learning resulting from experience can occur within networks, which can derive conclusions from a complex and seemingly unrelated set of information."""
#text = """'''Deep learning'''  (also known as '''deep structured learning'''  or '''hierarchical learning''') is part of a broader family of [[machine learning]] methods based on artificial neural networks. Learning can be [[Supervised learning|supervised]], [[Semi-supervised learning|semi-supervised]] or [[Unsupervised learning|unsupervised]].<ref name="BENGIO2012" /><ref name="SCHIDHUB" /><ref name="NatureBengio">{{cite journal |last1=Bengio |first1=Yoshua |last2=LeCun |first2= Yann| last3=Hinton | first3= Geoffrey|year=2015 |title=Deep Learning |journal=Nature |volume=521 |issue=7553 |pages=436–444 |doi=10.1038/nature14539 |pmid=26017442|bibcode=2015Natur.521..436L }}</ref>
#
#Deep learning architectures such as [[#Deep_neural_networks|deep neural network]]s, [[deep belief network]]s, [[recurrent neural networks]] and [[convolutional neural networks]] have been applied to fields including [[computer vision]], [[automatic speech recognition|speech recognition]], [[natural language processing]], audio recognition, social network filtering, [[machine translation]], [[bioinformatics]], [[drug design]], medical image analysis, material inspection and [[board game]] programs, where they have produced results comparable to and in some cases superior to human experts."""
#text = """'''''The Office''''' <!--DO NOT change to "was" as fictional works always remain in present tense regardless of completion-->is an American television [[sitcom]] that aired on [[NBC]] from March 24, 2005, to May 16, 2013, lasting nine seasons.<ref>{{cite web|url=http://www.thefutoncritic.com/showatch/office/ |title=Shows A-Z - The Office on NBC |website=The Futon Critic |accessdate=June 13, 2018}}</ref> It is an adaptation of the original [[BBC]] [[The Office (British TV series)|series of the same name]] and was adapted for American television by [[Greg Daniels]], a veteran writer for ''[[Saturday Night Live]]'', ''[[King of the Hill]]'', and ''[[The Simpsons]]''. It was co-produced by Daniels' [[Deedle-Dee Productions]], and [[Reveille Productions]] (later [[Endemol Shine North America|Shine America]]), in association with [[Universal Television]]. The original executive producers were Daniels, [[Howard Klein (television producer)|Howard Klein]], [[Ben Silverman]], [[Ricky Gervais]], and [[Stephen Merchant]], with numerous others being promoted in later seasons.
#
#The series depicts the everyday lives of office employees in the [[Scranton, Pennsylvania]] branch of the fictional [[Dunder Mifflin|Dunder Mifflin Paper Company]]. To [[Mockumentary|simulate the look of an actual documentary]], it was filmed in a [[single-camera setup]], without a [[studio audience]] or a [[laugh track]]. The series debuted on NBC as a [[midseason replacement]] and aired [[List of The Office (American TV series) episodes|201 episodes]] over the course of its run. ''The Office'' initially featured [[Steve Carell]], [[Rainn Wilson]], [[John Krasinski]], [[Jenna Fischer]], and [[B. J. Novak]] as the main cast; the series experienced [[List of The Office (American TV series) characters|numerous changes]] to its [[ensemble cast]] during its run. Notable stars outside the original main cast include [[Ed Helms]], [[Mindy Kaling]], [[Craig Robinson (actor)|Craig Robinson]], [[James Spader]], and [[Ellie Kemper]].
#"""
#text = """'''Apple Inc.''' is an American [[multinational corporation|multinational]] [[technology company]] headquartered in [[Cupertino, California]], that designs, develops, and sells [[consumer electronics]], [[software|computer software]], and [[online services]]. It is considered one of the [[Big Four tech companies]] along with [[Amazon (company)|Amazon]], [[Google]], and [[Facebook]].<ref>{{cite news |url= http://www.barrons.com/articles/ranking-the-big-four-internet-stocks-google-is-no-1-apple-comes-in-last-1503412102 |title= Ranking The Big Four Tech Stocks: Google Is No. 1, Apple Comes In Last|last=Rivas|first=Teresa|newspaper=[[Barron's (newspaper)|Barron's]] |language= en-US |access-date=December 27, 2018}}</ref><ref>{{Cite web|url=https://www.bloomberg.com/opinion/articles/2017-10-31/the-big-four-of-technology|title=The Big Four of Technology|last=Ritholtz|first=Barry|date=October 31, 2017|website=[[Bloomberg L.P.]]|access-date= December 27, 2018}}</ref>
#
#The company's [[computer hardware|hardware]] products include the [[iPhone]] smartphone, the [[iPad]] tablet computer, the [[Macintosh|Mac]] personal computer, the [[iPod]] portable media player, the [[Apple Watch]] smartwatch, the [[Apple TV]] digital media player, the [[AirPods]] wireless earbuds and the [[HomePod]] smart speaker. Apple's software includes the [[macOS]], [[iOS]], [[iPadOS]], [[watchOS]], and [[tvOS]] operating systems, the [[iTunes]] media player, the [[Safari (web browser)|Safari]] web browser, the [[Shazam (application)|Shazam]] acoustic fingerprint utility, and the [[iLife]] and [[iWork]] creativity and productivity suites, as well as professional applications like [[Final Cut Pro]], [[Logic Pro]], and [[Xcode]]. Its online services include the [[iTunes Store]], the [[App Store (iOS)|iOS App Store]], [[Mac App Store]], [[Apple Music]], [[Apple TV+]], [[iMessage]], and [[iCloud]]. Other services include [[Apple Store]], [[Genius Bar]], [[AppleCare]], [[Apple Pay]], [[Apple Pay Cash]], and [[Apple Card]].
#
#Apple was founded by [[Steve Jobs]], [[Steve Wozniak]], and [[Ronald Wayne]] in April 1976 to develop and sell Wozniak's [[Apple I]] personal computer, though Wayne sold his share back within 12 days."""
#text = """Microsoft was founded by [[Bill Gates]] and [[Paul Allen]] on April 4, 1975, to develop and sell [[BASIC]] [[Interpreter (computing)|interpreters]] for the [[Altair 8800]]. It rose to dominate the personal computer operating system market with [[MS-DOS]] in the mid-1980s, followed by [[Microsoft Windows]]. The company's 1986 [[initial public offering]] (IPO), and subsequent rise in its share price, created three billionaires and an estimated 12,000 millionaires among Microsoft employees. Since the 1990s, it has increasingly diversified from the [[operating system]] market and has made a number of [[List of mergers and acquisitions by Microsoft|corporate acquisitions]], their largest being the acquisition of [[LinkedIn]] for $26.2 billion in December 2016"""
#text = """The '''iPhone''' is a line of [[smartphone]]s designed and marketed by [[Apple Inc.]] All generations of the iPhone use Apple's [[iOS]] mobile operating system software. The [[iPhone (1st generation)|first-generation iPhone]] was released on June 29, 2007, and multiple new hardware iterations with new iOS releases have been released since.
#
#The [[user interface]] is built around the device's [[multi-touch]] screen, including a [[virtual keyboard]]. The iPhone has [[Wi-Fi]] and can connect to [[cellular network]]s. An iPhone can [[camera phone|take photos]], [[portable media player|play music]], send and receive [[email]], [[web browser|browse the web]], send and receive [[text messaging|text messages]], record notes, perform mathematical calculations, and receive [[visual voicemail]]. [[video camera|Shooting video]] also became a standard feature with the [[iPhone 3GS]]. Other functionality, such as video games, reference works, and social networking, can be enabled by downloading [[mobile app]]s. {{As of|2017|1}}, Apple's [[App Store (iOS)|App Store]] contained more than 2.2&nbsp;million applications available for the iPhone."""
#text = """Linux was originally developed for [[personal computer]]s based on the [[Intel x86]] architecture, but has since been [[porting|ported]] to more [[computer hardware platforms|platforms]] than any other operating system.<ref>{{cite news |author=Barry Levine |title=Linux' {{sic|22|th|nolink=yes}} Birthday Is Commemorated - Subtly - by Creator |url=http://www.cmswire.com/cms/information-management/linux-22th-birthday-is-commemorated-subtly-by-creator-022244.php |accessdate=May 10, 2015 |publisher=Simpler Media Group, Inc |date=August 26, 2013 |quote="Originally developed for Intel x86-based PCs, Torvalds' "hobby" has now been released for more hardware platforms than any other OS in history." |deadurl=no |archiveurl=https://web.archive.org/web/20150518155152/http://www.cmswire.com/cms/information-management/linux-22th-birthday-is-commemorated-subtly-by-creator-022244.php |archivedate=May 18, 2015  }}</ref> Linux is the leading operating system on [[server (computing)|servers]] and other  [[Big iron (computing)|big iron]] systems such as [[mainframe computer]]s, and the only OS used on [[TOP500]] [[supercomputer]]s (since November 2017, having gradually eliminated all competitors)."""
text = """PyTorch is an open source machine learning framework that is used by both researchers and developers to build, train, and deploy ML systems that solve many different complex challenges. The PyTorch team invites you to hack with the PyTorch community to build innovative, impactful models, applications and other projects that create positive impact for businesses or people. 

PyTorch is continually evolving to make it easier and faster for researchers to go from research exploration to production deployment. Some of the PyTorch capabilities and tools you can work with for the hackathon include:"""
text = """PyTorch is an open source machine learning library based on the Torch library,[1][2][3] used for applications such as computer vision and natural language processing.[4] It is primarily developed by Facebook's artificial intelligence research group.[5][6][7] It is free and open-source software released under the Modified BSD license. Although the Python interface is more polished and the primary focus of development, PyTorch also has a C++ frontend.[8] Furthermore, Uber's Pyro probabilistic programming language software uses PyTorch as a backend.[9]

PyTorch provides two high-level features:[10]

Tensor computing (like NumPy) with strong acceleration via graphics processing units (GPU)
Deep neural networks built on a tape-based autodiff system"""
text = """TensorFlow is Google Brain's second-generation system. Version 1.0.0 was released on February 11, 2017.[11] While the reference implementation runs on single devices, TensorFlow can run on multiple CPUs and GPUs (with optional CUDA and SYCL extensions for general-purpose computing on graphics processing units).[12] TensorFlow is available on 64-bit Linux, macOS, Windows, and mobile computing platforms including Android and iOS.

Its flexible architecture allows for the easy deployment of computation across a variety of platforms (CPUs, GPUs, TPUs), and from desktops to clusters of servers to mobile and edge devices.

TensorFlow computations are expressed as stateful dataflow graphs. The name TensorFlow derives from the operations that such neural networks perform on multidimensional data arrays, which are referred to as tensors. During the Google I/O Conference in June 2016, Jeff Dean stated that 1,500 repositories on GitHub mentioned TensorFlow, of which only 5 were from Google.[13]

In Jan 2018, Google announced TensorFlow 2.0.[14] In March 2018, Google announced TensorFlow.js version 1.0 for machine learning in JavaScript and TensorFlow Graphics for deep learning in computer graphics.[15][16]"""
text = """PHP: Hypertext Preprocessor (or simply PHP) is a general-purpose programming language originally designed for web development. It was originally created by Rasmus Lerdorf in 1994;[6] the PHP reference implementation is now produced by The PHP Group.[7] PHP originally stood for Personal Home Page,[6] but it now stands for the recursive initialism PHP: Hypertext Preprocessor.[8]

PHP code may be executed with a command line interface (CLI), embedded into HTML code, or used in combination with various web template systems, web content management systems, and web frameworks. PHP code is usually processed by a PHP interpreter implemented as a module in a web server or as a Common Gateway Interface (CGI) executable. The web server outputs the results of the interpreted and executed PHP code, which may be any type of data, such as generated HTML code or binary image data. PHP can be used for many programming tasks outside of the web context, such as standalone graphical applications[9] and robotic drone control.[10]

The standard PHP interpreter, powered by the Zend Engine, is free software released under the PHP License. PHP has been widely ported and can be deployed on most web servers on almost every operating system and platform, free of charge.[11]"""
text = """A new luxury condo rising high above SoMa in the heart of downtown San Francisco, 181 Fremont is a landmark residence. Two and three bedroom ultra-luxury residences with interiors designed by Orlando Diaz-Azcuy begin over 500’ feet in the sky and deliver breathtaking, panoramic views. Meticulously crafted kitchens and master baths boast the finest fixtures and finishes available for a condo. 181 Fremont features 6,500 square feet of amenities encompassing an entire floor including a piano bar, two lounges, fitness center, and wraparound observation terrace. Residents will enjoy unrivaled privacy, 24/7 lobby attendant, anticipatory concierge service, and underground valet parking."""
text = """An index is a specific structure that organizes a reference to your data that makes it easier to look up. In Postgres it is a copy of the item you wish to index combined with a reference to the actual data location. When accessing data, Postgres will either use some form of an index if it exists or a sequential scan. A sequential scan is when it searches over all of the data before returning the results.\nAdvantages and Disadvantages\nIndexes are great for accessing your data faster. In most cases adding an index to a column will allow you to query the data faster. However, the trade off is that for each index you have you will insert data at a slower pace. Essentially when you insert your data with an index it must\nwrite data to two places as well as maintain the sort on the index as you insert data. Certain indexes additionally will be more effective than others, such as indexes on numbers or timestamps (text is expensive)."""

text = """MINIX (from "mini-Unix") is a POSIX-compliant (since version 2.0),[3][4] Unix-like operating system based on a microkernel architecture.

Early versions of MINIX were created by Andrew S. Tanenbaum for educational purposes. Starting with MINIX 3, the primary aim of development shifted from education to the creation of a highly reliable and self-healing microkernel OS. MINIX is now developed as open-source software.

MINIX was first released in 1987, with its complete source code made available to universities for study in courses and research. It has been free and open-source software since it was re-licensed under the BSD license in April 2000.[5]
"""

if type(text) == str:
    text = text.encode('utf8')
maxlen = (2 * 1400) - 1
orig = text[:maxlen]
text = [str(c) if c != ord('\n') else '<eos>' for c in text]
text = text
text = [dictionary.word2idx[c] for c in text]
print(text)

input = torch.rand(1, 1).mul(ntokens).long()
print(input.shape)

input = torch.Tensor(text).view(-1, 1).long()
if args.cuda:
    input = input.cuda()
logits, hidden, mems = model(input[:-1, :], hidden, mems=mems, return_h=False)
input = input[-1:, :]
# TODO: We lose a token here as we predict one, update the memory, but don't add it to our generated text

def produce_vocab_logits(head_weight, head_bias, hiddens):
    head_res = torch.nn.functional.linear(hiddens, head_weight, bias=head_bias)
    #softmaxed_head_res = torch.nn.functional.log_softmax(head_res, dim=-1)
    #softmaxed_head_res = F.softmax(head_res, dim=-1)
    return head_res

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

with open(args.outf, 'w') as outf:
    outf.write(str(orig.decode('utf8')))
    outf.write('||||')

    for i in range(args.words):
        with torch.no_grad():
            logits, hidden, new_mems = model(input, hidden, mems=mems, return_h=False)
        # TODO: What if we want to start with no history?
        magic_mem = []
        for ma, mb in zip(mems, new_mems):
            magic_mem.append(torch.cat([ma, mb], dim=0)[-maxlen:])
        mems = magic_mem
        output = produce_vocab_logits(model.decoder.weight, model.decoder.bias, logits) / args.temperature
        #output = top_k_top_p_filtering(output.view(-1), top_p=0.91).view(*output.shape)
        word_weights = F.softmax(output, dim=-1).squeeze()
        #word_weights = output.squeeze().data.div(args.temperature).exp().cpu()
        word_idx = torch.multinomial(word_weights, num_samples=1)[0]
        input.data.fill_(word_idx)
        word = dictionary.idx2word[word_idx]

        #outf.write(word + ('\n' if i % 20 == 19 else ' '))
        outf.write(chr(int(word)) if word != '<eos>' else '\n')

        if i % args.log_interval == 0:
            print('| Generated {}/{} words'.format(i, args.words))
            print('|| Memory: {}'.format(None if mems is None else mems[0].shape))

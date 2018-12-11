#!/usr/bin/env python3
#
# ./ocr.py : Perform optical character recognition, usage:
#     ./ocr.py train-image-file.png train-text.txt test-image-file.png
# 
# Authors: (insert names here)
# (based on skeleton code by D. Crandall, Oct 2018)
#
"""
Please refer readme for all the test cases run.

Calculations of initial state distribution, Transition and Emission probabilities:

    Here , Hidden States are the actual textual letters 
    Observed variables are the character images
    When a letter transition is not already in the trained dictionary we substitiute by small probability
     math.log(1.7976931348623157e-308)
    
    Initial State Distribution:
    
    
    The probabilities are namely P(S1), P(Si+1|Si),
    and P(Wi|Si)
    
    For P(S1)(namely the initial state distribution) that is the prior of "letter" at position 1,
    we have collected all the letters of the first position in each sentence and put it as a key in a dictionary(being startingProbs{} ) 
    of logarithm of probabilities. 
    Hence we have letters as keys and their corresponding logarithm of probabilities of them appearing in the first position. 
    
    For priors to be considered in positions other than first, I have collected all the 'letters' appearing in the 
    positions other than the first and found out the probabilities. This froms my dictionary allSimpleLetterProbs{}.
    This was done to make a distinction in distributions for POS according to their positions so as to get better accuracy.
    
    Transition probabilities:
        
        P(Si+1|Si): (Required for HMM)
        Since P(Si+1|Si) = P(Si+1 , Si)/P(Si), We have found the probability of two 'letters' appearing in sequence,
        that is P(Si+1 , Si), for example [h-e], P(h,e) is the probability of h-e pair appearing and 
        is stored as the transition probabilities(in the form of log(P)), transitionProbs{} dictionary.
        And to calculate -->> log(P(Si+1|Si)) = log(P(Si+1 , Si))-log(P(Si)),
        For example, log(P('h'/'e')) = log(P('h' , 'e'))-log(P('e')) = transitionProbs["h_e"]-allSimplePOSProbs["e"]
        
        
    Emission probabilities:
        
        P(Observed Character Image|letter) : 
        Final : I have taken F - Scores my mathicng the image at the pixel level , I have the follwing table
        simialr to true positives, true negatives, false positives and false negatives
        
        A lit pixel is where there is '*'
        A unlit pixel is where there is ' '
        
                                     ActuallyLit   |  Actually Unlit
                     
               Test Pixel Lit       litMatched     | unlitUnmatched
               Test Pixel Unlit     litUnmacthed   | unlitMatched                

        For F Score i calculate precision and recall for lit pixel as matching a lit pixel is more important
        precision = number of litMatched pixels / number of litMatched +number of unlitUnmatched
        recall = number of litMatched/ number of 'litUnmatched'+  number of 'litMatched'

        F Score is the harmonic mean of preciaion and recall 
        i.e emissionGivenActLetter = (((1/precision) + (1/recall))/2)^(-1)
            emissionGivenActLetter = emissionGivenActLetter**TOTAL_PIXELS
        now this FScore is the accuracy of the match of the test image with the train image 
        or the probability of each pixel at a letter matching the test image's pixel  
        Hence I raise it to the power of Total Number of pixels assuming a Naive Bayes assumtion for the pixels
        to get the overall emission probaility.
        Also I scale the emission probability so that they match the level of transiotion and starting
        as the this was done as I obeserved the emission was around -300 in log probability
        and trainsitions and prrioes were around -20. 
        Also I handled the case where the litMatched was zero , when there is a space and assign it 1 such that 
        I avoid zero error
        
        Tried methods:
            
            I had explored the possibility of Bayes net assumption for pixels such that the pixels in a row are
            as chain and are independetn of the pixels in the next row.
            However it didnt work as expected and hence I had dropped this plan.
            The functions,


Simple Naive Bayes Approach:
    
    In this we make an naive assumption that each letter at a give position is independent of any other positions.
    That is, P(S1,S2,S3,S4,....,Sn/O1,O2,O3,...,On) = P(O1/S1)*P(S1)*P(O2/S2)*P(S2)....*P(On/Sn)*P(Sn),
    so at every ith position, we find out the "si" leading to the highest probabilty of P(Si = si/W).
    and output the sequence of the letters.
    
    
HMM Model By Viterbi(Fig 1(a)):
    
    Since each state is dependent on the previous state we would need to calculate the sequence "s1,s2,s3,s4,....,sn" 
    such that the probability P(Si = si|O) leads to maximum.
    That is, P(S1,S2,S3,S4,....,Sn/O1,O2,O3,...,On) = P(O1/S1)*P(S1)*P(O2/S2)*P(S2/S1)....*P(On/Sn)*P(Sn/Sn-1),
    so at every ith position, we find out the "si" leading to the highest probabilty of P(Si = si/W).
    and output the sequence of the letters.
       
    For every state at position t, we find the max of previous states' probability times the transition probability from the 
    previous state to current state and multiply with emission probability of the current state.
    We store this probability in a dictionary dictOfPostions[position][letter] in the form of log(P), now after 
    we calculate the probability at each position for each state, to backtrack we append in the form of string 
    
     maxPaths[position][letter] = maxPaths[position-1][prevStateLeadingToMax] + "=>" + POS, 
     where POS is the current part of speech.(POS at position "t"), such that we dont have to actually traverse or backtrack
     through the entire trace, rather at the last position find the letter that had the maximum probability and pull the string
     that  maxPaths[LastPosition][maxLetter] had. Hence saving the requirement of return loop.
            


"""

from PIL import Image, ImageDraw, ImageFont
import math
import sys
import collections as col
import itertools as itr
CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25
SMALL_PROB = math.log(1.7976931348623157e-308)
LIT_PIXEL = '*'
UNLIT_PIXEL = ' '
EmissionProbModel = 'Naive Bayes'
TOTAL_PIXELS = CHARACTER_WIDTH*CHARACTER_HEIGHT


def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
#    print (im.size)
#    print (int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH)
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
    return result

def load_training_letters(fname):
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }





def load_data(filename):
        
        ##returns list of sentences as each sentence is ( (words),(tags))
        listOfPOS = ('det','noun','adj','verb','adp','.','adv','conj','prt','pron','num','x')
        
        AllSentencesAsLetters= [] ## [((words),(corresponding tags)), (sentence 2)]
        file = open(filename, 'r');
        for line in file:
            ##avoidifing the file with POS tags
            data = tuple([w for w in line.split()])
            
            sentence = ' '.join(data[0::2])
   
#            tags = data[1::2]
            Letters = tuple(sentence)
            
            
            AllSentencesAsLetters.append(Letters) #isolating sentence as well
            Letters = tuple(sentence.upper())
            AllSentencesAsLetters.append(Letters)
            
            
        file.close()
        return AllSentencesAsLetters ##rerutns list of tuples as characters in the sentece  


#####
# main program
    
    
    
def train(AllSentencesAsLetters):
    
    ###Initial letter probability####
    AllLetter1List = [characters[0] for characters in AllSentencesAsLetters]
        
    Total_Letter1 = len(AllLetter1List)
    ProbOfLetter1AsDict = dict(col.Counter(AllLetter1List)) 
    ProbOfLetter1AsDict = {letter: math.log(frequency/Total_Letter1) for letter,frequency in ProbOfLetter1AsDict.items()}
    
    
    ###rest letters probabilities
    
    AllButLetter1List = list(itr.chain.from_iterable([characters[1:] for characters in AllSentencesAsLetters]))
    Total_LetterButL1 = len(AllButLetter1List)
    ProbOfLetter_AllAsDict = dict(col.Counter(AllButLetter1List)) 
    ProbOfLetter_AllAsDict = {letter : math.log(frequency/Total_LetterButL1) for letter, frequency in ProbOfLetter_AllAsDict.items()} 
    
    
    ######transition probabilities
    
    seqLetters = []
    for sentence in AllSentencesAsLetters:
         ##can be combined in list comprehension but would lose readability
         
         seqLetters += [ letter + "_" + nextLetter for letter,nextLetter in zip(sentence,sentence[1:])]
     
    TotalNumSequences = len(seqLetters)
        
    ProbOfLetterSeqAsDict = dict(col.Counter(seqLetters))
    ProbOfLetterSeqAsDict = {letterSeq: math.log(frequency/TotalNumSequences) for letterSeq,frequency in ProbOfLetterSeqAsDict.items()}
    
    
    
    
    return ProbOfLetter1AsDict,ProbOfLetter_AllAsDict,ProbOfLetterSeqAsDict

def emission_prob(testImageRep , actualLetter, dictOfCorrectRep ):
    
    """
    
    I have taken F - Scores mathicng the image at the pixel level , I have the follwing table
    simialr to true positives, true negatives, false positives and false negatives
    The FScore are raised to power of total pixels to know the overall probability
     Takes in testImageRep in pixels i.e list of strings
     actualLetter the letter being checked for 
     dictOfCorrectRep : Dictionary of train_letters i.e correct representaion of letters in pixel form
     Returns  Probability



                       ActuallyLit   |  ActuallyUnlit
         
   TestLit            litMatched     | unlitUnmatched
   TestPixelUnlit     litUnmacthed   | unlitMatched                 
     
    """
        
    global EmissionProbModel
    m = 0.05    
    
    if EmissionProbModel == 'Naive Bayes':
    
    
        dictOfMatches = col.defaultdict(int)
        TotalLitPixels = 0
        TotalUnlitPixels = 0
        
        
        for testRow , actualRow in zip(testImageRep,dictOfCorrectRep[actualLetter]):
            for pixelTest, pixelActual in zip(testRow,actualRow):
                
                
                if pixelActual == LIT_PIXEL:
                    TotalLitPixels += 1
                    if pixelTest == pixelActual:
                        dictOfMatches['litMatched'] += 1
         
                    else:
                        dictOfMatches['litUnmatched'] += 1    
                
                elif pixelActual == UNLIT_PIXEL:
                    TotalUnlitPixels += 1
                    if pixelTest == pixelActual:
                        dictOfMatches['unlitMatched'] += 1
                    else:
                        dictOfMatches['unlitUnmatched'] += 1
        
        
        try: 
            
            
            ##handling when lit pixels are less than two i.e a space
            if TotalLitPixels <= 5:
                dictOfMatches['litMatched'] = 2
                
            precision = dictOfMatches['litMatched']/(dictOfMatches['litMatched']+dictOfMatches['unlitUnmatched'])
            recall = dictOfMatches['litMatched']/(dictOfMatches['litUnmatched']+dictOfMatches['litMatched'])
#           
            emissionGivenActLetter = (((1/precision) + (1/recall))/2)**(-1)
            emissionGivenActLetter = emissionGivenActLetter**TOTAL_PIXELS
#            emissionGivenActLetter
#            emissionGivenActLetter = Probability
            if emissionGivenActLetter == 0.0:
                emissionGivenActLetter = 1.7976931348623157e-308
        except :
            emissionGivenActLetter = 1.7976931348623157e-308
        #    print(matchedCount)
        #   emissionGivenActLetter = matchedCount/(CHARACTER_HEIGHT*CHARACTER_WIDTH)
        return emissionGivenActLetter
    
    
    
    """
    Calculatin of the emission probabilities based on the Bayes net assumption
    
    """
    if EmissionProbModel == 'Bayes Net':
        
        logProb = 0
        for testRow , actualRow in zip(testImageRep,dictOfCorrectRep[actualLetter]):
        
            for position, (testPixel , prevActualPixel, ActualPixel) in enumerate(zip(testRow,[actualRow[0]]+list(actualRow),actualRow)): ##(e,e,f,g,h),(e,f,g,h)
            
                if position == 0:
                    ###possible cases of the distributions
                    if testPixel == ActualPixel and ActualPixel == LIT_PIXEL :   
                        logProb += math.log(1-m) + dictOfPixelProbs[actualLetter][LIT_PIXEL]
                        
                    if testPixel == ActualPixel and ActualPixel == NON_LIT_PIXEL :   
                        logProb += math.log(1-m) + dictOfPixelProbs[actualLetter][NON_LIT_PIXEL]
                        
                    if testPixel != ActualPixel and ActualPixel == LIT_PIXEL :   
                        logProb += math.log(m) + dictOfPixelProbs[actualLetter][LIT_PIXEL]
                    
                    if testPixel != ActualPixel and ActualPixel == NON_LIT_PIXEL :   
                        logProb += math.log(m) + dictOfPixelProbs[actualLetter][NON_LIT_PIXEL]
                    
                    
                else :
                    
                    if testPixel == ActualPixel :   
                        logProb += math.log(1-m) + dictOfPixelTransitionProbs[actualLetter][prevActualPixel+"_"+ActualPixel]
                        
                    
                    if testPixel != ActualPixel  :   
                        logProb += math.log(m) + dictOfPixelTransitionProbs[actualLetter][prevActualPixel+"_"+ActualPixel]
                    
                    
                    
                    
                    
            return logProb  
            

def getEmissionProbDict(testLetter):
    
#    emissionProbDict = {queryletter: emission_prob(testLetter,queryletter,train_letters) for queryletter in train_letters}
    emissionProbDict = {queryletter: math.log(emission_prob(testLetter,queryletter,train_letters))/12 for queryletter in train_letters}
#    TotalEmission = sum(emissionProbDict.values())
#    emissionProbDict = {queryletter : math.log(value/TotalEmission) for queryletter,value in emissionProbDict.items()}
    
    return emissionProbDict


def train_emissionModelParams(dictOfCorrectRep):
    
    
    dictOfTransitionProbs = col.defaultdict(dict)
    dictOfPixelProbs = col.defaultdict(dict)
    
    
    for letter in dictOfCorrectRep:
    
        
        seqPixels = []
        for pixelRow in dictOfCorrectRep[letter]:
            
            ##transition probabilities
            seqPixels.extend([ pixel + "_" + nextPixel for pixel,nextPixel in zip(pixelRow,pixelRow[1:])])
        
        
        ####starting probabilities priors
        collapsedListPixels = list(itr.chain.from_iterable(dictOfCorrectRep[letter]))
        
        totalPixels = len(collapsedListPixels)
        
        dictOfPixelProbs[letter] = dict(col.Counter(collapsedListPixels))
        
        dictOfTransitionProbs[letter] = dict(col.Counter(seqPixels))
        totalTransistions = sum(dictOfTransitionProbs[letter].values())
        
        
        dictOfPixelProbs[letter] = {key: math.log(value/totalPixels) for key,value in dictOfPixelProbs[letter].items()}
        dictOfTransitionProbs[letter]  = {key: math.log(value/totalTransistions) for key,value in dictOfTransitionProbs[letter].items()}
    
    return dictOfPixelProbs,dictOfTransitionProbs ##as logarithmic
    
    

(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]

#train_img_fname = "C:/Users/ntihish/Documents/IUB/Elem of AI/asiignment 3/prgadugu-skandag-chhshar-a3/part2/courier-train.png"
train_letters = load_training_letters(train_img_fname)
#test_img_fname = "C:/Users/ntihish/Documents/IUB/Elem of AI/asiignment 3/prgadugu-skandag-chhshar-a3/part2/test-3-0.png"
test_letters = load_letters(test_img_fname)
 #list of strings representing letters



    
def Solve_hmm_viterbi(test_letters,train_letters):

   ##checck if trained 
   lastPosition = len(test_letters)-1
   ##sentence is passed as tuple of words
   maxPaths = col.defaultdict(dict)
   dictOfPostions = col.defaultdict(dict) ##postionons and letter probabilities
   for position,testLetter in enumerate(test_letters) :
       #for severy postion
       
#       
       emissionProbDict = getEmissionProbDict(testLetter)
#       
       
       for queryletter in train_letters :
        #for eveery state at a postion
           if position == 0 :
               
                   
               dictOfPostions[position][queryletter] = allSimpleLetterProbs.get(queryletter,SMALL_PROB) \
               + emissionProbDict.get(queryletter,SMALL_PROB)
               
               maxPaths[position][queryletter] = queryletter
               
               
               
           else:
               #max((value, prevPOS))
           
               (logProbability,prevStateLeadingToMax)  = \
               max(( (dictOfPostions[position-1][prevLetter]+ transitionProbs.get(prevLetter + "_" + queryletter, SMALL_PROB) - allSimpleLetterProbs.get(prevLetter,SMALL_PROB), prevLetter)\
                    for prevLetter in train_letters))

                 
                   
             
               dictOfPostions[position][queryletter] = logProbability + emissionProbDict.get(queryletter,SMALL_PROB)
               
               maxPaths[position][queryletter] = maxPaths[position-1][prevStateLeadingToMax] + "=>" + queryletter
               ##back tracking in one line :)
   maxLastProb,lastMaxLetter = max(((probability,lastLetter) for lastLetter,probability in dictOfPostions[lastPosition].items()))
 
   return maxPaths[lastPosition][lastMaxLetter].replace("=>",'')



def solve_By_Simple(test_letters,train_letters):
    maxLettersList = []
    """
    In this we make an naive assumption that each letter at a give position is independent of any other positions.
    That is, P(S1,S2,S3,S4,....,Sn/O1,O2,O3,...,On) = P(O1/S1)*P(S1)*P(O2/S2)*P(S2)....*P(On/Sn)*P(Sn),
    so at every ith position, we find out the "si" leading to the highest probabilty of P(Si = si/W).
    and output the sequence of the letters.
    """   
    
    for position,testLetter in enumerate(test_letters):
        maxLetter = ''
        maxProb = -float("inf")
        emissionProbDict = getEmissionProbDict(testLetter)
        for queryletter in train_letters :
            if position == 0 :
                
                currentProb = emissionProbDict.get(queryletter,SMALL_PROB) + startingProbs.get(queryletter,SMALL_PROB)
            
            else:
                
                currentProb = emissionProbDict.get(queryletter,SMALL_PROB) + allSimpleLetterProbs.get(queryletter,SMALL_PROB)
            
            if currentProb > maxProb:
                maxProb = currentProb
                maxLetter = queryletter
        maxLettersList.append(maxLetter)
        
                
    return ''.join(maxLettersList)




##load and train the tranistions and staritng probabilieites
data = load_data(train_txt_fname)

startingProbs,allSimpleLetterProbs,transitionProbs = train(data)



print("Simple: ", solve_By_Simple(test_letters,train_letters))
print("Viterbi: ", Solve_hmm_viterbi(test_letters,train_letters))
print("Final answer:")
print(Solve_hmm_viterbi(test_letters,train_letters) )

if __name__ == '__main__':
#    myList = []
#    for i in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' ":
#        myList.append((emission_prob(test_letters[0],i,train_letters),i))
#    min(myList[:-1])
#    allSimpleLetterProbs.get('S',SMALL_PROB)
#    startingProbs.get('T',SMALL_PROB) + emission_prob(test_letters[0],'S',train_letters)
#    transitionProbs.get('S_U',0)
##    startingProbs.get('T',SMALL_PROB) + emission_prob(test_letters[0],'T',train_letters)
##    max(allSimpleLetterProbs.values())
##    max(allSimpleLetterProbs, key=allSimpleLetterProbs.get)
##    allSimpleLetterProbs[' ']
#    max(startingProbs.values())
    
#    test_img_fname = "C:/Users/ntihish/Documents/IUB/Elem of AI/asiignment 3/prgadugu-skandag-chhshar-a3/part2/test-0-0.png"
#    test_letters = load_letters(test_img_fname)
#    math.log(emission_prob(test_letters[0],'S',train_letters))

#    transitionProbs.get('1_t',0)
#    startingProbs.get('1',SMALL_PROB)
#    for i in train_letters:
#        print(i ,"  ",  startingProbs.get(i,SMALL_PROB))
#        print(transitionProbs.get(i+'_U',0))
#        print(i , "  ", math.log(emission_prob(test_letters[0],i,train_letters)))
#        
        
#    print("Simple: ", solve_By_Simple(test_letters,train_letters))
#    print("Viterbi: ", Solve_hmm_viterbi(test_letters,train_letters))
#    
#    file = open("test-strings.txt", 'r');
#    for i,line in enumerate(file):
#        print("Actual :" , line)
#        test_img_fname = "C:/Users/ntihish/Documents/IUB/Elem of AI/asiignment 3/prgadugu-skandag-chhshar-a3/part2/test-"+ str(i) +"-0.png"
#        test_letters = load_letters(test_img_fname)
#        print("Simple : ",solve_By_Simple(test_letters,train_letters))
#        print("Viterbi : " , Solve_hmm_viterbi(test_letters,train_letters))
#        print("---------------------------------------------------------  ")
    pass    
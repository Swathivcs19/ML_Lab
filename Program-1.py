{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[['sky', 'airtemp', 'humidity', 'wind', 'water', 'forcast', 'enjoysport'], ['sunny', 'warm', 'normal', 'strong', 'warm', 'same', 'yes'], ['sunny', 'warm', 'high', 'strong', 'warm', 'same', 'yes'], ['rainy', 'cold', 'high', 'strong', 'warm', 'change', 'no'], ['sunny', 'warm', 'high', 'strong', 'cool', 'change', 'yes']]\n",
      "\n",
      " The total number of training instances are:  5\n",
      "\n",
      "The initial hypothesis is: \n",
      "['0', '0', '0', '0', '0', '0']\n",
      "\n",
      " The hypothesis for the training instance 1 is:\n",
      " ['0', '0', '0', '0', '0', '0']\n",
      "\n",
      " The hypothesis for the training instance 2 is:\n",
      " ['sunny', 'warm', 'normal', 'strong', 'warm', 'same']\n",
      "\n",
      " The hypothesis for the training instance 3 is:\n",
      " ['sunny', 'warm', '?', 'strong', 'warm', 'same']\n",
      "\n",
      " The hypothesis for the training instance 4 is:\n",
      " ['sunny', 'warm', '?', 'strong', 'warm', 'same']\n",
      "\n",
      " The hypothesis for the training instance 5 is:\n",
      " ['sunny', 'warm', '?', 'strong', '?', '?']\n",
      "\n",
      " The maximally specific hypothesis for the training is\n",
      "['sunny', 'warm', '?', 'strong', '?', '?']\n"
     ]
    }
   ],
   "source": [
    "import csv\n",
    "a=[]\n",
    "with open('enjoysport.csv','r') as csvfile:\n",
    "    for row in csv.reader(csvfile):\n",
    "        a.append(row)\n",
    "    print(a)\n",
    "print(\"\\n The total number of training instances are: \",len(a))\n",
    "\n",
    "num_attribute=len(a[0])-1\n",
    "print(\"\\nThe initial hypothesis is: \")\n",
    "hypothesis = ['0']*num_attribute\n",
    "print(hypothesis)\n",
    "for i in range(0, len(a)):\n",
    "    if a[i][num_attribute] == 'yes':\n",
    "        for j in range(0, num_attribute):\n",
    "            if hypothesis[j] == '0' or hypothesis[j] == a[i][j]:\n",
    "                hypothesis[j] = a[i][j]\n",
    "            else:\n",
    "                hypothesis[j] ='?'\n",
    "    print(\"\\n The hypothesis for the training instance {} is:\\n\" .format(i+1),hypothesis)\n",
    "print(\"\\n The maximally specific hypothesis for the training is\")\n",
    "print(hypothesis)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

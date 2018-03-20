INSTRUCTIONS:
1. Created and tested in windows OS.
2. Requires Python 3 installed.
3. Requires nltk packages installed. open a bash, write 'import nltk' and 'nltk.download()' selecting download all packages from the menu.
4. nltk.stem, nltk.tokenize and nltk.corpus are required.
5. Throw the 'hashtags.py' in the same folder with the test documents. 
	Make sure you have write 'w' rights within that folder as the exported files are created within.
6. run from a command window showing in that location, 'python hashtags.py', for the default 20 top words to be printed within 'hashtags_out.txt'
You can run 'python hashtags.py #', where # is a number of your choice, for the # number of top words to be printed.
You can run 'python hashtags.py # csv' for the # number of top words to be exported in a .csv file, so that a more reusable and technically useful form of the data is exported. (Pandas library will instantly read this csv as a proper matrix.)
(Note that instead of csv, CSV/.csv/Csv/excel/Excel are also acceptable, as read from an array of acceptable terms CodeLine (cl) 176.

NOTES:
1. cl 130: more stopwords can be added in case of a different language.
2. cl 134: documents can be added in that array to be read. As 'doc5.txt' needed 'utf8' encoding, it was added. If another document needs another encoding it can easily be added with minor implementation.
3. The writeTXT and writeCSV functions were created for better presentation and usefullness accordingly.
4. Duplicate tokens within the same sentence are removed so that repeating words to increase "occurrence count" is avoided.
5. The merge_obj(list, list2) function aims to merge the output of two documents. Therefore higher scalability is achieved, not allowing large arrays of the saved terms to be passed from the one function to the other and be parsed every time.

NEXT STEPS: 
1. Allow from command argument to stem or not the tokens.
2. Add popularity metrics of words. Implementing metrics from large corpus and dividing by the term frequency that is currently calculated, can lead to more interesting results.

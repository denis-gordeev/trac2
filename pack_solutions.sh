for task in a b
do
    for lang in eng hin iben
    do
        zip "final_${lang}_${task}.zip" "final_${lang}_${task}.csv" description.txt
    done
done
zip final_eng_a.zip final_eng_a.csv description.txt
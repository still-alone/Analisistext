package org.example;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.SparkConf;
import org.tartarus.snowball.ext.RussianStemmer;
import scala.Tuple2;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

public class TextAnalysis {

    public static void main(String[] args) throws IOException {
        // Настройка Spark
        SparkConf conf = new SparkConf()
                .setAppName("TextAnalysis")
                .setMaster("local[*]")
                .set("spark.security.manager", "NONE"); // Отключаем Security Manager
        JavaSparkContext sc = new JavaSparkContext(conf);

        // Чтение файла из ресурсов
        String inputPath = "src/main/resources/text.txt";
        JavaRDD<String> textFile = sc.textFile(inputPath);

        // 1. Очистка текста (удаление стоп-слов, исправление проблем с данными)
        Set<String> stopWords = new HashSet<>(Files.readAllLines(Paths.get("src/main/resources/stopwords.txt")));
        JavaRDD<String> cleanedText = textFile
                .flatMap(line -> Arrays.asList(line.toLowerCase().split("[^а-яА-Яa-zA-Z0-9]+")).iterator())
                .filter(word -> !stopWords.contains(word) && !word.isEmpty());

        // 2. Подсчет количества слов (WordCount)
        JavaRDD<Tuple2<String, Integer>> wordCounts = cleanedText
                .mapToPair(word -> new Tuple2<>(word, 1))
                .reduceByKey(Integer::sum)
                .map(tuple -> new Tuple2<>(tuple._1, tuple._2));

        long count = wordCounts
                .map(Tuple2::_2)
                .reduce(Integer::sum);

        System.out.println("Кол-во слов в тексте: " + count);

        // Сортировка для топ-50 самых частых и редких слов
        List<Tuple2<String, Integer>> sortedWordCounts = wordCounts.collect()
                .stream()
                .sorted(Comparator.comparingInt(Tuple2::_2))
                .collect(Collectors.toList());

        List<Tuple2<String, Integer>> top50MostCommon = sortedWordCounts.stream()
                .sorted((a, b) -> b._2 - a._2)
                .limit(50)
                .collect(Collectors.toList());

        List<Tuple2<String, Integer>> top50LeastCommon = sortedWordCounts.stream()
                .limit(50)
                .collect(Collectors.toList());

        System.out.println("Top 50 Most Common Words:");
        top50MostCommon.forEach(tuple -> System.out.println(tuple._1 + ": " + tuple._2));

        System.out.println("Top 50 Least Common Words:");
        top50LeastCommon.forEach(tuple -> System.out.println(tuple._1 + ": " + tuple._2));

        // 4. Стемминг текста
        JavaRDD<String> stemmedWords = cleanedText.map(TextAnalysis::stemWord);

        // Подсчет количества слов после стемминга
        JavaRDD<Tuple2<String, Integer>> stemmedWordCounts = stemmedWords
                .mapToPair(word -> new Tuple2<>(word, 1))
                .reduceByKey(Integer::sum)
                .map(tuple -> new Tuple2<>(tuple._1, tuple._2));

        long countAfterStemming = stemmedWordCounts
                .map(Tuple2::_2)
                .reduce(Integer::sum);

        System.out.println("Кол-во слов после стемминга: " + countAfterStemming);

        // Сортировка для топ-50 самых частых и редких слов после стемминга
        List<Tuple2<String, Integer>> sortedStemmedWordCounts = stemmedWordCounts.collect()
                .stream()
                .sorted(Comparator.comparingInt(Tuple2::_2))
                .collect(Collectors.toList());

        List<Tuple2<String, Integer>> top50MostCommonStemmed = sortedStemmedWordCounts.stream()
                .sorted((a, b) -> b._2 - a._2)
                .limit(50)
                .collect(Collectors.toList());

        List<Tuple2<String, Integer>> top50LeastCommonStemmed = sortedStemmedWordCounts.stream()
                .limit(50)
                .collect(Collectors.toList());

        System.out.println("Top 50 Most Common Words After Stemming:");
        top50MostCommonStemmed.forEach(tuple -> System.out.println(tuple._1 + ": " + tuple._2));

        System.out.println("Top 50 Least Common Words After Stemming:");
        top50LeastCommonStemmed.forEach(tuple -> System.out.println(tuple._1 + ": " + tuple._2));

        sc.close();
    }

    // Рабочий стеммер для русского языка
    private static String stemWord(String word) {
        RussianStemmer stemmer = new RussianStemmer();
        stemmer.setCurrent(word);
        if (stemmer.stem()) {
            return stemmer.getCurrent();
        }
        return word;
    }
}

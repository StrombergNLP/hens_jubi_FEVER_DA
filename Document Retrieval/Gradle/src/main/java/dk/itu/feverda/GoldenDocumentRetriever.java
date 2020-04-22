package dk.itu.feverda;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import opennlp.tools.postag.POSModel;
import opennlp.tools.postag.POSTaggerME;
import opennlp.tools.sentdetect.SentenceDetectorME;
import opennlp.tools.sentdetect.SentenceModel;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.IntPoint;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.analysis.da.DanishAnalyzer;

public class GoldenDocumentRetriever {
    public static void main(String[] args) throws Exception {
        // Building the wiki index
        System.out.println("Building index...");
        DanishAnalyzer analyzer = new DanishAnalyzer();
        Path indexPath = Files.createTempDirectory("wikiIndex");
        Directory directory = FSDirectory.open(indexPath);
        IndexWriterConfig config = new IndexWriterConfig(analyzer);
        IndexWriter iWriter = new IndexWriter(directory, config);

        // Build index from Wikipedia data
        String wikiPath = "./src/main/resources/dawiki-latest-pages-articles-parsed.csv";
        Scanner wikiSc = new Scanner(new File(wikiPath));
        wikiSc.nextLine(); // skip header
        while(wikiSc.hasNextLine()) {
            String[] line = wikiSc.nextLine().split(";");
            String evidence = line[1];
            String title = line[3];
            addDoc(iWriter, evidence, title);
        }
        wikiSc.close();
        iWriter.close();

        DirectoryReader iReader = DirectoryReader.open(directory);
        IndexSearcher iSearcher = new IndexSearcher(iReader);

        // Read our annotations
        System.out.println("Querying with claims...");
        String testPath = args.length > 0 ? args[0] : "../Data Generation/Final Dataset/annotations_dev.tsv";
        Scanner testSc = new Scanner(new File(testPath));
        testSc.nextLine(); // skip header

        DateTimeFormatter dtf = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH-mm-ss");
        LocalDateTime now = LocalDateTime.now();
        String outFileName = "out_" + dtf.format(now) + ".jsonl";
        FileWriter outWriter = new FileWriter(outFileName);
        // outWriter.append("claim\tevidence\tlabel\n");

        // Set up models for tokenizing and POS tagging
        InputStream tokenModelIn = new FileInputStream("da-token.bin");
        TokenizerModel tokenModel = new TokenizerModel(tokenModelIn);
        TokenizerME tokenizer = new TokenizerME(tokenModel);
        InputStream POSModelIn = new FileInputStream("da-pos-perceptron.bin");
        POSModel POSModel = new POSModel(POSModelIn);
        POSTaggerME POSTagger = new POSTaggerME(POSModel);

        while(testSc.hasNextLine()) {
            String[] line = testSc.nextLine().split("\t");
            String claim = line[1];
            String[] tokenizedClaim = tokenizer.tokenize(claim);
            String[] claimPOS = POSTagger.tag(tokenizedClaim);
            String nounClaim = "";
            for (int i = 0; i < tokenizedClaim.length; i++) {
                String token = tokenizedClaim[i];
                String tag = claimPOS[i];
                if (tag.startsWith("N")) nounClaim += " " + token;
            }
            String escapedClaim = QueryParser.escape(nounClaim.trim());
            String label = line[2];

            // Searching the index
            QueryParser parser = new QueryParser("evidence", analyzer);
            Query query = parser.parse(escapedClaim);
            int k = 5;
            ScoreDoc[] hits = iSearcher.search(query, k).scoreDocs;

            // System.out.println(String.format("\nClaim: %s\nResults:", claim));
            List<String> evidence = new ArrayList<>();
            for (int i = 0; i < hits.length; i++) {
                Document hitDoc = iSearcher.doc(hits[i].doc);
                evidence.add(hitDoc.get("evidence"));
                // System.out.println(String.format("\t%d: %s", i + 1, hitDoc.get("evidence")));
            }

            String joinedEvidence = String.join(" ", evidence); // Concatenate to one string

            // Split into sentences
            InputStream modelIn = new FileInputStream("da-sent.bin");
            SentenceModel sentModel = new SentenceModel(modelIn);
            SentenceDetectorME sentenceDetector = new SentenceDetectorME(sentModel);
            String[] sentences = sentenceDetector.sentDetect(joinedEvidence);

            // Building the sentence index
            DanishAnalyzer sentAnalyzer = new DanishAnalyzer();
            Path sentIndexPath = Files.createTempDirectory("sentIndex");
            Directory sentDirectory = FSDirectory.open(sentIndexPath);
            IndexWriterConfig sentConfig = new IndexWriterConfig(sentAnalyzer);
            IndexWriter sentIWriter = new IndexWriter(sentDirectory, sentConfig);

            // System.out.println("\n" + claim);
            for(int i = 0; i < sentences.length; i++) {
                addSent(sentIWriter, sentences[i], i);
                // System.out.println(sentences[i]);
            }
            sentIWriter.close();

            // Searching the sent index
            DirectoryReader sentIReader = DirectoryReader.open(sentDirectory);
            IndexSearcher sentISearcher = new IndexSearcher(sentIReader);
            QueryParser sentParser = new QueryParser("sentence", sentAnalyzer);
            Query sentQuery = sentParser.parse(escapedClaim);
            int sentK = 2;
            ScoreDoc[] sentHits = sentISearcher.search(sentQuery, sentK).scoreDocs;

            List<String> selectedSentences = new ArrayList<>();
            for (int j = 0; j < sentHits.length; j++) {
                Document hitDoc = sentISearcher.doc(sentHits[j].doc);
                selectedSentences.add(hitDoc.get("sentence"));
            }

            String joinedSelectedEvidence = String.join(" ", selectedSentences); // Concatenate to one string
            // outWriter.append(claim).append("\t").append(joinedSelectedEvidence).append("\t").append(label).append("\n");
            outWriter.append("{\"claim\":\"").append(claim.replace("\"", "\\\""))
                .append("\",\"evidence\":\"").append(joinedSelectedEvidence.replace("\"", "\\\""))
                .append("\",\"label\":\"").append(label).append("\"}\n");

            sentIReader.close();
            sentDirectory.close();
            IOUtils.rm(sentIndexPath);
        }
        
        // Cleanup
        outWriter.close();
        testSc.close();
        iReader.close();
        directory.close();
        IOUtils.rm(indexPath);
        System.out.println("Done.");
    }

    private static void addDoc(IndexWriter w, String evidence, String title) throws IOException {
        Document doc = new Document();
        doc.add(new Field("evidence", evidence, TextField.TYPE_STORED));
        doc.add(new Field("title", title, StringField.TYPE_STORED));
        w.addDocument(doc);
    }

    private static void addSent(IndexWriter w, String sentence, int index) throws IOException {
        Document doc = new Document();
        doc.add(new Field("sentence", sentence, TextField.TYPE_STORED));
        doc.add(new IntPoint("index", index));
        w.addDocument(doc);
    }
}
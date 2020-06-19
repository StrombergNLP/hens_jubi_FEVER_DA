package com.function;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import com.microsoft.azure.functions.ExecutionContext;

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
    public static String retrieve(String claim, ExecutionContext context) throws Exception {
        // Wiki index
        ClassLoader classLoader = Thread.currentThread().getContextClassLoader();
        DanishAnalyzer analyzer = new DanishAnalyzer();
        // Path indexPath = Paths.get("wikiIndex").toAbsolutePath();
        Path indexPath = Paths.get("D:\\home\\data\\wikiIndex");
        Directory directory;
        // Check if wiki index exists from previous build
        if (Files.exists(indexPath)) {
            context.getLogger().info("Reusing previous wiki index.");
            directory = FSDirectory.open(indexPath);
        } else {
            context.getLogger().info("Creating wikiIndex directory at " + indexPath);
            indexPath = Files.createDirectory(indexPath);
            context.getLogger().info("Opening wiki index...");
            directory = FSDirectory.open(indexPath);
            IndexWriterConfig config = new IndexWriterConfig(analyzer);
            IndexWriter iWriter = new IndexWriter(directory, config);

            // Build index from Wikipedia data
            context.getLogger().info("Loading wiki tsv...");
            String wikiPath = "dawiki-latest-pages-articles-parsed.tsv";
            Scanner wikiSc = new Scanner(classLoader.getResourceAsStream(wikiPath), "UTF-8");
            wikiSc.nextLine(); // skip header
            List<String> invalidLines = new ArrayList<>();
            context.getLogger().info("Building wiki index...");
            while(wikiSc.hasNextLine()) {
                String line = wikiSc.nextLine();
                // context.getLogger().info(line);
                String[] splitline = line.split("\t");
                if (splitline.length == 4) {
                    String evidence = splitline[1];
                    String title = splitline[3];
                    addDoc(iWriter, evidence, title);
                } else {
                    invalidLines.add(line);
                }   
            }
            context.getLogger().info("Unable to parse " + invalidLines.size() + " lines.");
            // for (String line : invalidLines) {
            //     context.getLogger().info("INVALID: " + line);
            // }
            wikiSc.close();
            iWriter.close();
            context.getLogger().info("Building wiki index complete.");
        }

        DirectoryReader iReader = DirectoryReader.open(directory);
        IndexSearcher iSearcher = new IndexSearcher(iReader);

        // Set up models for tokenizing and POS tagging
        InputStream tokenModelIn = classLoader.getResourceAsStream("da-token.bin");
        TokenizerModel tokenModel = new TokenizerModel(tokenModelIn);
        TokenizerME tokenizer = new TokenizerME(tokenModel);
        InputStream POSModelIn = classLoader.getResourceAsStream("da-pos-perceptron.bin");
        POSModel POSModel = new POSModel(POSModelIn);
        POSTaggerME POSTagger = new POSTaggerME(POSModel);

        String evidence = "";
        if (claim != "") {
            context.getLogger().info(String.format("Received claim '%s' from args.", claim));
            evidence = getEvidence(claim, tokenizer, iSearcher, analyzer, POSTagger, classLoader, 1, 4);
        } else {
            context.getLogger().info("Cannot parse arguments!");
        }
        
        // Cleanup
        iReader.close();
        directory.close();
        // IOUtils.rm(indexPath);
        context.getLogger().info("Done.");

        return evidence;
    }

    private static String getEvidence(String claim, TokenizerME tokenizer, IndexSearcher iSearcher, DanishAnalyzer analyzer, POSTaggerME POSTagger, ClassLoader classLoader, int k, int sentK) throws Exception { 
        String[] tokenizedClaim = tokenizer.tokenize(claim);
        String[] claimPOS = POSTagger.tag(tokenizedClaim);
        String nounClaim = "";
        for (int i = 0; i < tokenizedClaim.length; i++) {
            String token = tokenizedClaim[i];
            String tag = claimPOS[i];
            if (tag.startsWith("N")) nounClaim += " " + token;
        }
        String escapedClaim = QueryParser.escape(nounClaim.trim());

        // Searching the index
        QueryParser parser = new QueryParser("evidence", analyzer);
        Query query = parser.parse(escapedClaim);
        ScoreDoc[] hits = iSearcher.search(query, sentK).scoreDocs;

        // context.getLogger().info(String.format("\nClaim: %s\nResults:", claim));
        List<String> evidence = new ArrayList<>();
        for (int i = 0; i < hits.length; i++) {
            Document hitDoc = iSearcher.doc(hits[i].doc);
            evidence.add(hitDoc.get("evidence"));
            // context.getLogger().info(String.format("\t%d: %s", i + 1, hitDoc.get("evidence")));
        }

        String joinedEvidence = String.join(" ", evidence); // Concatenate to one string

        // Split into sentences
        InputStream modelIn = classLoader.getResourceAsStream("da-sent.bin");
        SentenceModel sentModel = new SentenceModel(modelIn);
        SentenceDetectorME sentenceDetector = new SentenceDetectorME(sentModel);
        String[] sentences = sentenceDetector.sentDetect(joinedEvidence);

        // Building the sentence index
        DanishAnalyzer sentAnalyzer = new DanishAnalyzer();
        Path sentIndexPath = Files.createTempDirectory("sentIndex");
        Directory sentDirectory = FSDirectory.open(sentIndexPath);
        IndexWriterConfig sentConfig = new IndexWriterConfig(sentAnalyzer);
        IndexWriter sentIWriter = new IndexWriter(sentDirectory, sentConfig);

        // context.getLogger().info("\n" + claim);
        for(int i = 0; i < sentences.length; i++) {
            addSent(sentIWriter, sentences[i], i);
            // context.getLogger().info(sentences[i]);
        }
        sentIWriter.close();

        // Searching the sent index
        DirectoryReader sentIReader = DirectoryReader.open(sentDirectory);
        IndexSearcher sentISearcher = new IndexSearcher(sentIReader);
        QueryParser sentParser = new QueryParser("sentence", sentAnalyzer);
        Query sentQuery = sentParser.parse(escapedClaim);
        ScoreDoc[] sentHits = sentISearcher.search(sentQuery, sentK).scoreDocs;

        List<String> selectedSentences = new ArrayList<>();
        for (int j = 0; j < sentHits.length; j++) {
            Document hitDoc = sentISearcher.doc(sentHits[j].doc);
            selectedSentences.add(hitDoc.get("sentence"));
        }

        String joinedSelectedEvidence = String.join(" ", selectedSentences); // Concatenate to one string

        sentIReader.close();
        sentDirectory.close();
        IOUtils.rm(sentIndexPath);

        return joinedSelectedEvidence;
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
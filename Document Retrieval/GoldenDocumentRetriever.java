import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import org.apache.lucene.analysis.da.DanishAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
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

public class LuceneTest {
    public static void main(String[] args) throws Exception {

        // Building the index
        System.out.println("Building index...");
        DanishAnalyzer analyzer = new DanishAnalyzer();
        Path indexPath = Files.createTempDirectory("tempIndex");
        Directory directory = FSDirectory.open(indexPath);
        IndexWriterConfig config = new IndexWriterConfig(analyzer);
        IndexWriter iWriter = new IndexWriter(directory, config);

        // Build index from Wikipedia data
        String wikiPath = "../Data Generation/CommonData/dawiki-latest-pages-articles-parsed.csv";
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
        String testPath = "../Data Generation/Final Dataset/annotations_test.tsv";
        Scanner testSc = new Scanner(new File(testPath));
        testSc.nextLine(); // skip header

        FileWriter csvWriter = new FileWriter("annotations_test_retrieved_evidence.tsv");
        csvWriter.append("claim\tevidence\tlabel\n");

        while(testSc.hasNextLine()) {
            String[] line = testSc.nextLine().split("\t");
            String claim = line[1];
            String label = line[2];

            // Searching the index
            QueryParser parser = new QueryParser("evidence", analyzer);
            Query query = parser.parse(claim);
            int k = 3;
            ScoreDoc[] hits = iSearcher.search(query, k).scoreDocs;

            System.out.println(String.format("\nClaim: %s\nResults:", claim));
            List<String> evidence = new ArrayList<>();
            for (int i = 0; i < hits.length; i++) {
                Document hitDoc = iSearcher.doc(hits[i].doc);
                evidence.add(hitDoc.get("evidence"));
                System.out.println(String.format("\t%d: %s", i + 1, hitDoc.get("evidence")));
            }

            String joinedEvidence = String.join(" .", evidence); // Concatenate to one string
            csvWriter.append(claim).append("\t").append(joinedEvidence).append("\t").append(label).append("\n");
        }
        
        // Cleanup
        csvWriter.close();
        testSc.close();
        iReader.close();
        directory.close();
        IOUtils.rm(indexPath);
    }

    private static void addDoc(IndexWriter w, String evidence, String title) throws IOException {
        Document doc = new Document();
        doc.add(new Field("evidence", evidence, TextField.TYPE_STORED));
        doc.add(new Field("title", title, StringField.TYPE_STORED));
        w.addDocument(doc);
    }
}
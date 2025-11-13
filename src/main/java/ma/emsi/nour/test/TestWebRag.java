package ma.emsi.nour.test;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import dev.langchain4j.rag.query.router.DefaultQueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;
import java.util.List;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class TestWebRag {

    interface Assistant {
        String chat(String message);
    }

    public static void main(String[] args) {

        configureLogger();

        String geminiKey = System.getenv("GEMINI_KEY");
        String tavilyKey = System.getenv("TAVILY_KEY");

        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(geminiKey)
                .modelName("gemini-2.5-flash")
                .temperature(0.1)
                .logRequestsAndResponses(true)
                .build();

        EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
        ingest("rag.pdf", store);

        ContentRetriever pdfRetriever =
                EmbeddingStoreContentRetriever.builder()
                        .embeddingStore(store)
                        .embeddingModel(new AllMiniLmL6V2EmbeddingModel())
                        .maxResults(2)
                        .minScore(0.5)
                        .build();

        TavilyWebSearchEngine webSearchEngine = TavilyWebSearchEngine.builder()
                .apiKey(tavilyKey)
                .build();

        ContentRetriever webRetriever =
                WebSearchContentRetriever.builder()
                        .webSearchEngine(webSearchEngine)
                        .maxResults(3)
                        .build();

        var router = new DefaultQueryRouter(List.of(pdfRetriever, webRetriever));

        RetrievalAugmentor augmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(router)
                .build();

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .retrievalAugmentor(augmentor)
                .build();

        conversationAvec(assistant);
    }

    private static void ingest(String resource, EmbeddingStore<TextSegment> store) {
        Document doc = ClassPathDocumentLoader.loadDocument(resource, new ApacheTikaDocumentParser());
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);

        List<TextSegment> segments = splitter.split(doc);

        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        var embeddings = embeddingModel.embedAll(segments).content();

        store.addAll(embeddings, segments);
    }

    private static void configureLogger() {
        Logger log = Logger.getLogger("dev.langchain4j");
        log.setLevel(Level.FINE);
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        log.addHandler(handler);
    }

    private static void conversationAvec(Assistant assistant) {
        Scanner sc = new Scanner(System.in);
        while (true) {
            System.out.println("==================================================");
            System.out.println("Posez une question (ou 'fin' pour arrêter) :");
            String q = sc.nextLine();
            if ("fin".equalsIgnoreCase(q)) break;
            String r = assistant.chat(q);
            System.out.println("Assistant : " + r);
        }
    }
}

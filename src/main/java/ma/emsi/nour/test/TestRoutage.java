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
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.util.*;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

import java.util.List;
import java.util.Scanner;

public class TestRoutage {

    public interface Assistant {
        String chat(String userMessage);
    }

    public static void main(String[] args) {

        configureLogger();

        String key = System.getenv("GEMINI_KEY");

        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(key)
                .modelName("gemini-2.5-flash")
                .temperature(0.2)
                .logRequestsAndResponses(true)
                .build();


        EmbeddingStore<TextSegment> storeIA = new InMemoryEmbeddingStore<>();
        EmbeddingStore<TextSegment> storeCuisine = new InMemoryEmbeddingStore<>();

        ingestDocument("rag.pdf", storeIA);
        ingestDocument("cuisine.txt", storeCuisine);


        ContentRetriever retrieverIA = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(storeIA)
                .embeddingModel(new AllMiniLmL6V2EmbeddingModel())
                .maxResults(2)
                .minScore(0.5)
                .build();

        ContentRetriever retrieverCuisine = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(storeCuisine)
                .embeddingModel(new AllMiniLmL6V2EmbeddingModel())
                .maxResults(2)
                .minScore(0.5)
                .build();

        Map<ContentRetriever, String> descriptions = new HashMap<>();
        descriptions.put(retrieverIA, "Contient des informations sur l'intelligence artificielle, le RAG, les LLMs.");
        descriptions.put(retrieverCuisine, "Contient des recettes de cuisine, des ingrédients.");

        QueryRouter router = new LanguageModelQueryRouter(model, descriptions);

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

    private static void ingestDocument(String resource, EmbeddingStore<TextSegment> store) {
        Document doc = ClassPathDocumentLoader.loadDocument(
                resource,
                new ApacheTikaDocumentParser()
        );

        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);
        List<TextSegment> segments = splitter.split(doc);

        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        var response = embeddingModel.embedAll(segments);
        store.addAll(response.content(), segments);
    }

    private static void conversationAvec(Assistant assistant) {
        try (Scanner scanner = new Scanner(System.in)) {
            while (true) {
                System.out.println("==================================================");
                System.out.println("Posez une question (ou 'fin' pour arrêter) :");
                String q = scanner.nextLine();

                if ("fin".equalsIgnoreCase(q)) break;
                if (q.isBlank()) continue;

                System.out.println("Assistant : " + assistant.chat(q));
            }
        }
    }

    private static void configureLogger() {
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE);

        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
    }
}
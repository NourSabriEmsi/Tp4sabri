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
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.util.*;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.Scanner;

public class TestPasDeRag {

    public interface Assistant {
        String chat(String message);
    }

    public static void main(String[] args) {

        configureLogger();

        String key = System.getenv("GEMINI_KEY");

        ChatModel modelChat = GoogleAiGeminiChatModel.builder()
                .apiKey(key)
                .modelName("gemini-2.5-flash")
                .temperature(0.2)
                .logRequestsAndResponses(true)
                .build();

        EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
        ingest("rag.pdf", store);

        ContentRetriever contentRetriever =
                EmbeddingStoreContentRetriever.builder()
                        .embeddingStore(store)
                        .embeddingModel(new AllMiniLmL6V2EmbeddingModel())
                        .maxResults(2)
                        .minScore(0.5)
                        .build();

        QueryRouter router = new QueryRouter() {

            PromptTemplate template = PromptTemplate.from(
                    "Est-ce que la requête suivante concerne l'intelligence artificielle, le RAG ou le fine-tuning ?\n" +
                            "Question : '{{question}}'\n" +
                            "Réponds uniquement par : 'oui', 'non' ou 'peut-être'."
            );

            @Override
            public List<ContentRetriever> route(Query query) {

                String prompt = template.apply(Map.of("question", query.text())).text();

                String answer = modelChat.chat(prompt).toLowerCase().trim();

                System.out.println("----- Décision du LM (route) -----");
                System.out.println("Question utilisateur : " + query.text());
                System.out.println("Réponse du LM : " + answer);

                if (answer.contains("non")) {
                    return Collections.emptyList();
                }
                return Collections.singletonList(contentRetriever);
            }
        };

        RetrievalAugmentor augmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(router)
                .build();

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(modelChat)
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
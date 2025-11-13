package ma.emsi.nour;

public class Main {
    public static void main(String[] args) {
        System.out.printf("Hello and welcome!");
        for (int i = 1; i <= 5; i++) {
            System.out.println("i = " + i);
        }       String tavilyKey = System.getenv("TAVILY_KEY");

        System.out.println("TAVILY_KEY = " + tavilyKey);

    }
}
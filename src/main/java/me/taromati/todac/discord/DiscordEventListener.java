package me.taromati.todac.discord;

import lombok.RequiredArgsConstructor;
import me.taromati.todac.discord.config.DiscordConfigProperties;
import net.dv8tion.jda.api.events.message.MessageReceivedEvent;
import net.dv8tion.jda.api.hooks.ListenerAdapter;
import org.springframework.stereotype.Component;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

@RequiredArgsConstructor
@Component
public class DiscordEventListener extends ListenerAdapter {
    private final DiscordConfigProperties discordConfigProperties;

    private final ExecutorService executorService = Executors.newVirtualThreadPerTaskExecutor();

    @Override
    public void onMessageReceived(MessageReceivedEvent event) {
        if (event.getChannel().getName().equals(discordConfigProperties.getChannelName()) == false) {
            return;
        } else if (event.getAuthor().isBot()) {
            return;
        }

        String message = event.getMessage().getContentRaw();

        CompletableFuture.runAsync(() -> {
            try {
                StringBuilder responseBuilder = new StringBuilder();
                ProcessBuilder pb = new ProcessBuilder(List.of("./venv/bin/python3", "./ai.py", message));
                Process p = pb.start();

                try (BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        System.out.println("Python script output: " + line);
                        responseBuilder.append(line).append("\n");
                    }
                }
                int exitCode = p.waitFor();
                System.out.println("Process exited with code: " + exitCode);

                event.getChannel().sendMessage(responseBuilder.toString()).queue();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }, executorService);

        System.out.printf("[%s] %#s: %s\n",
                event.getChannel(),
                event.getAuthor(),
                event.getMessage().getContentDisplay());
    }
}

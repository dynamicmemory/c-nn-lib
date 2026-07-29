#include <stdio.h>
#include <string.h>
#include <netdb.h>
#include <sys/socket.h>

void generate_body(char body[], char prompt[], uint32_t buf_size);
void http_request(char http_req[], char body[], uint32_t buf_size);

int main(void) {
    // Set up connect to local model
    char host[] = "0.0.0.0";
    char port[] = "11434";

    struct addrinfo hints, *addr;
    memset(&hints, 0, sizeof(struct addrinfo));
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_family = AF_UNSPEC;
    if (getaddrinfo(host, port, &hints, &addr) != 0) {
        fprintf(stderr, "getaddrinfo failed");
        return 1;
    }

    int server = socket(addr->ai_family, addr->ai_socktype, addr->ai_protocol);
    if (server < 0) { 
        fprintf(stderr, "socket failed");
        return 1;
    }

    if (connect(server, addr->ai_addr, addr->ai_addrlen) < 0) {
        fprintf(stderr, "connect failed");
        return 1;
    }
    freeaddrinfo(addr);

    // Construct prompt + http request
    char prompt[] = "Explain bitcoin in one sentence";
    char body[4096];
    generate_body(body, prompt, sizeof(body));

    char http_req[4096];
    http_request(http_req, body, sizeof(http_req)); 

    // Send and recieve message from llm
    send(server, http_req, sizeof(http_req), 0);

    while (1) {
        char res[4096];
        char *p;
        int recieve = recv(server, res, sizeof(res), 0);
        // Find the response and print it to the terminal char by char
        if ((p=strstr(res, "\"response\":\""))) {
            p+=12;

            while (*p && *p != '"')
                putchar(*p++);
        }

        fflush(stdout);
        // Exit loop on response finished
        if ((p=strstr(res, "\"done\":true"))) {
            printf("\n");
            break;
        }
    }

    return 0;
}

void generate_body(char body[], char prompt[], uint32_t buf_size) {
    snprintf(body, buf_size, 
        "{\"model\": \"llama3.1:8B\","
        "\"prompt\": \"%s\","
        "\"stream\": true}",
        prompt);
}

void http_request(char request[], char body[], uint32_t buf_size) {
    snprintf(request, buf_size,
        "POST /api/generate HTTP/1.1\r\n"
        "Host: localhost:11434\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %zu\r\n"
        "\r\n" 
        "%s",
        strlen(body), body);
    
}

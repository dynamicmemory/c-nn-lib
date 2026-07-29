#include <string>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cstring>
#include <iostream>

int main(void) {
    const char host[] = "127.0.0.1";
    const char port[] = "11434";

    struct addrinfo hints, *addr;
    ::memset(&hints, 0, sizeof(addrinfo));
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_family = AF_UNSPEC;
    
    int status = ::getaddrinfo(host, port, &hints, &addr);
    if (status != 0) {
        std::cout << "getaddrinfo failed\n";
        return 1;
    }
   
    int socket = ::socket(addr->ai_family, addr->ai_socktype, addr->ai_protocol);
    if (socket < 0) {
        std::cout << "socket failed\n";
        return 1;
    }
    
    int connect = ::connect(socket, addr->ai_addr, addr->ai_addrlen);
    if (connect < 0) {
        std::cout << "connect failed\n";
        return 1;
    }
    ::freeaddrinfo(addr);
    
    std::string body = 
        R"({"model": "llama3.1:8B", 
        "prompt": "Explain bitcoin in one sentence", 
        "stream": false})";

    std::string http_req = 
        "POST /api/generate HTTP/1.1\r\n"
        "Host: localhost:11434\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: " + std::to_string(body.size()) + "\r\n"
        "\r\n" + 
        body;

    ssize_t send = ::send(socket, http_req.data(), http_req.size(), 0); 
    std::cout << "Size sent: " << send << "\n";

    std::string res;
    char buff[4096];
    while (res.find("\r\n\r\n") == std::string::npos) {
        ssize_t recieve = ::recv(socket, buff, sizeof(buff), 0);    

        if (recieve <= 0)
            break;

        res.append(buff, recieve);
    }
    std::cout << res << std::endl;

    return 0;
}

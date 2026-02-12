/*
 * dpdk_custom_packet_sender.c
 * DPDK packet sender that reads custom packets from a binary file
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_mbuf.h>
#include <rte_cycles.h>

#define BURST_SIZE 32
#define MEMPOOL_CACHE_SIZE 256
#define NUM_MBUFS 8191

static struct rte_mempool *mbuf_pool = NULL;

int main(int argc, char *argv[]) {
    int ret;
    uint16_t port_id = 0;
    FILE *fp;
    uint32_t num_packets;
    uint64_t total_sent = 0;
    uint64_t start_tsc, end_tsc;
    double elapsed_sec, pps, gbps;
    
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <packet_file>\n", argv[0]);
        return 1;
    }
    
    // Initialize DPDK EAL with minimal args
    char *eal_argv[] = {"packet_sender", "-l", "0", "--proc-type=primary", NULL};
    ret = rte_eal_init(4, eal_argv);
    if (ret < 0) {
        fprintf(stderr, "Error: EAL initialization failed\n");
        return 1;
    }
    
    // Create mbuf pool
    mbuf_pool = rte_pktmbuf_pool_create(
        "MBUF_POOL",
        NUM_MBUFS,
        MEMPOOL_CACHE_SIZE,
        0,
        RTE_MBUF_DEFAULT_BUF_SIZE,
        rte_socket_id()
    );
    
    if (mbuf_pool == NULL) {
        fprintf(stderr, "Error: Cannot create mbuf pool\n");
        return 1;
    }
    
    // Get device info
    struct rte_eth_dev_info dev_info;
    ret = rte_eth_dev_info_get(port_id, &dev_info);
    if (ret != 0) {
        fprintf(stderr, "Error: Cannot get device info\n");
        return 1;
    }
    
    // Configure port - TX only for mlx4 compatibility
    struct rte_eth_conf port_conf = {0};
    port_conf.rxmode.mq_mode = RTE_ETH_MQ_RX_NONE;
    port_conf.txmode.mq_mode = RTE_ETH_MQ_TX_NONE;
    port_conf.rxmode.offloads = 0;
    port_conf.txmode.offloads = 0;
    
    // Configure with 0 RX queues, 1 TX queue (TX-only!)
    ret = rte_eth_dev_configure(port_id, 0, 1, &port_conf);
    if (ret < 0) {
        fprintf(stderr, "Error: Cannot configure device (code %d)\n", ret);
        return 1;
    }
    
    // Setup TX queue
    struct rte_eth_txconf txconf = {0};
    txconf.offloads = 0;
    ret = rte_eth_tx_queue_setup(port_id, 0, 512, rte_eth_dev_socket_id(port_id), &txconf);
    if (ret < 0) {
        fprintf(stderr, "Error: Cannot setup TX queue (code %d)\n", ret);
        return 1;
    }
    
    // Start device
    ret = rte_eth_dev_start(port_id);
    if (ret < 0) {
        fprintf(stderr, "Error: Cannot start device (code %d)\n", ret);
        return 1;
    }
    
    // Enable promiscuous mode
    ret = rte_eth_promiscuous_enable(port_id);
    if (ret != 0) {
        fprintf(stderr, "Warning: Cannot enable promiscuous mode (code %d)\n", ret);
    }
    
    printf("DPDK initialized successfully!\n");
    
    // Open packet file
    fp = fopen(argv[1], "rb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open packet file %s\n", argv[1]);
        return 1;
    }
    
    // Read number of packets
    if (fread(&num_packets, 4, 1, fp) != 1) {
        fprintf(stderr, "Error: Cannot read packet count\n");
        fclose(fp);
        return 1;
    }
    
    printf("Sending %u packets...\n", num_packets);
    
    // Allocate packet burst
    struct rte_mbuf *bufs[BURST_SIZE];
    
    start_tsc = rte_rdtsc();
    
    // Send packets
    while (total_sent < num_packets) {
        uint16_t nb_tx = (num_packets - total_sent) < BURST_SIZE ? 
                         (num_packets - total_sent) : BURST_SIZE;
        
        // Allocate mbufs
        if (rte_pktmbuf_alloc_bulk(mbuf_pool, bufs, nb_tx) != 0) {
            fprintf(stderr, "Error: Failed to allocate mbufs\n");
            break;
        }
        
        // Read and fill packets
        for (int i = 0; i < nb_tx; i++) {
            uint16_t pkt_len;
            if (fread(&pkt_len, 2, 1, fp) != 1) {
                fprintf(stderr, "Error: Cannot read packet length\n");
                // Free already allocated mbufs
                for (int j = 0; j <= i; j++) {
                    rte_pktmbuf_free(bufs[j]);
                }
                goto cleanup;
            }
            
            uint8_t *data = rte_pktmbuf_mtod(bufs[i], uint8_t *);
            if (fread(data, 1, pkt_len, fp) != pkt_len) {
                fprintf(stderr, "Error: Cannot read packet data\n");
                for (int j = 0; j <= i; j++) {
                    rte_pktmbuf_free(bufs[j]);
                }
                goto cleanup;
            }
            
            bufs[i]->data_len = pkt_len;
            bufs[i]->pkt_len = pkt_len;
        }
        
        // Send burst
        uint16_t nb_sent = rte_eth_tx_burst(port_id, 0, bufs, nb_tx);
        
        // Free unsent packets
        if (nb_sent < nb_tx) {
            for (int i = nb_sent; i < nb_tx; i++) {
                rte_pktmbuf_free(bufs[i]);
            }
        }
        
        total_sent += nb_sent;
    }
    
cleanup:
    end_tsc = rte_rdtsc();
    fclose(fp);
    
    // Calculate performance
    uint64_t tsc_hz = rte_get_tsc_hz();
    elapsed_sec = (double)(end_tsc - start_tsc) / tsc_hz;
    pps = total_sent / elapsed_sec;
    gbps = (pps * 64 * 8) / 1e9;  // Assume avg 64 bytes
    
    printf("\nResults:\n");
    printf("  Packets sent: %"PRIu64"\n", total_sent);
    printf("  Time:         %.3f ms\n", elapsed_sec * 1000);
    printf("  PPS:          %.0f (%.1fM pps)\n", pps, pps / 1e6);
    printf("  Throughput:   %.2f Gbps\n", gbps);
    
    // Cleanup
    rte_eth_dev_stop(port_id);
    rte_eal_cleanup();
    
    return 0;
}



/*
 * dpdk_packet_sender.c
 * Simple DPDK packet sender for benchmarking
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

#define PACKET_SIZE 64
#define BURST_SIZE 32
#define MEMPOOL_CACHE_SIZE 256
#define NUM_MBUFS 8191

static struct rte_mempool *mbuf_pool = NULL;

static void create_test_packet(struct rte_mbuf *mbuf) {
    uint8_t *data = rte_pktmbuf_mtod(mbuf, uint8_t *);
    
    // Ethernet header (dummy)
    memset(data, 0xFF, 6);      // Dst MAC (broadcast)
    memset(data + 6, 0x00, 6);  // Src MAC
    data[12] = 0x08;             // EtherType
    data[13] = 0x00;             // IPv4
    
    // Padding
    memset(data + 14, 0x00, PACKET_SIZE - 14);
    
    mbuf->data_len = PACKET_SIZE;
    mbuf->pkt_len = PACKET_SIZE;
}

int main(int argc, char *argv[]) {
    int ret;
    uint16_t port_id = 0;
    uint32_t num_packets;
    uint64_t total_sent = 0;
    uint64_t start_tsc, end_tsc;
    double elapsed_sec, pps, gbps;
    
    if (argc < 2) {
        printf("Usage: %s <num_packets>\n", argv[0]);
        return 1;
    }
    
    num_packets = atoi(argv[1]);
    
    // Initialize DPDK EAL
    ret = rte_eal_init(argc, argv);
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
    
    // Get device info to check capabilities
    struct rte_eth_dev_info dev_info;
    ret = rte_eth_dev_info_get(port_id, &dev_info);
    if (ret != 0) {
        fprintf(stderr, "Error: Cannot get device info\n");
        return 1;
    }
    
    // Configure port - TX only for mlx4 compatibility (no RX = no flow rules!)
    struct rte_eth_conf port_conf = {0};
    // Disable all multiqueue modes
    port_conf.rxmode.mq_mode = RTE_ETH_MQ_RX_NONE;
    port_conf.txmode.mq_mode = RTE_ETH_MQ_TX_NONE;
    // Disable all offloads
    port_conf.rxmode.offloads = 0;
    port_conf.txmode.offloads = 0;
    
    // Configure with 0 RX queues, 1 TX queue (TX-only mode!)
    ret = rte_eth_dev_configure(port_id, 0, 1, &port_conf);
    if (ret < 0) {
        fprintf(stderr, "Error: Cannot configure device (code %d)\n", ret);
        return 1;
    }
    
    // NO RX queue setup - this avoids flow rule issues!
    
    // Setup TX queue with minimal config
    struct rte_eth_txconf txconf = {0};
    txconf.offloads = 0;  // No offloads
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
    
    // Enable promiscuous mode (often needed for mlx4)
    ret = rte_eth_promiscuous_enable(port_id);
    if (ret != 0) {
        fprintf(stderr, "Warning: Cannot enable promiscuous mode (code %d)\n", ret);
        // Don't fail - continue anyway
    }
    
    printf("DPDK initialized successfully!\n");
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
        
        // Fill packets
        for (int i = 0; i < nb_tx; i++) {
            create_test_packet(bufs[i]);
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
    
    end_tsc = rte_rdtsc();
    
    // Calculate performance
    uint64_t tsc_hz = rte_get_tsc_hz();
    elapsed_sec = (double)(end_tsc - start_tsc) / tsc_hz;
    pps = total_sent / elapsed_sec;
    gbps = (pps * PACKET_SIZE * 8) / 1e9;
    
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

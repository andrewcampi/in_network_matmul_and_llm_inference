#!/bin/bash
# Deep dive into telemetry configuration and firewall counter support

SWITCH_IP="${1:-10.10.10.55}"
SSH_KEY="/home/multiplex/.ssh/id_rsa"

echo "========================================================================"
echo "TELEMETRY CONFIGURATION DEEP DIVE - ${SWITCH_IP}"
echo "========================================================================"
echo ""

echo "1. Show current telemetry/analytics configuration:"
ssh -i ${SSH_KEY} root@${SWITCH_IP} "cli -c 'show configuration services | display set'"
echo ""

echo "2. Check for extension-service (gRPC server) configuration:"
ssh -i ${SSH_KEY} root@${SWITCH_IP} "cli -c 'show configuration system services extension-service | display set'"
echo ""

echo "3. List ALL available sensor resources (what can we monitor?):"
ssh -i ${SSH_KEY} root@${SWITCH_IP} "cli -c 'show agent sensors'"
echo ""

echo "4. Check if gRPC server is running and what port:"
ssh -i ${SSH_KEY} root@${SWITCH_IP} "netstat -an | grep -E 'LISTEN.*:(50051|32767|32768)'"
echo ""

echo "5. Check for JTI/gRPC processes:"
ssh -i ${SSH_KEY} root@${SWITCH_IP} "ps aux | grep -i jti"
echo ""

echo "6. Show network-agent configuration (streaming telemetry service):"
ssh -i ${SSH_KEY} root@${SWITCH_IP} "cli -c 'show configuration system services netconf | display set'"
echo ""

echo "7. Check if we can configure new sensors (test command):"
ssh -i ${SSH_KEY} root@${SWITCH_IP} "cli -c 'configure; show services analytics sensor ?; rollback 0'"
echo ""

echo "8. Check available firewall sensor paths:"
ssh -i ${SSH_KEY} root@${SWITCH_IP} "cli -c 'show agent sensors' | grep -i firewall"
echo ""

echo "9. Test if we can read firewall counters via RPC:"
ssh -i ${SSH_KEY} root@${SWITCH_IP} "cli -c 'show firewall filter layer0_filter | display xml'"
echo ""

echo "10. Check extension toolkit availability (for gRPC server setup):"
ssh -i ${SSH_KEY} root@${SWITCH_IP} "ls -la /var/db/scripts/jet/"
echo ""

echo "========================================================================"
echo "NEXT: Check what resources are available for custom sensors"
echo "========================================================================"
echo ""

echo "11. Full sensor list with resource paths:"
ssh -i ${SSH_KEY} root@${SWITCH_IP} "cli -c 'show agent sensors' | grep Resource"
echo ""

echo "12. Check if OpenConfig paths are available (alternative):"
ssh -i ${SSH_KEY} root@${SWITCH_IP} "cli -c 'show agent sensors' | grep -E '/interfaces/|/system/'"
echo ""

echo "========================================================================"
echo "DONE"
echo "========================================================================"


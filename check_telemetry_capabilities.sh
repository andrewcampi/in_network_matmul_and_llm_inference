#!/bin/bash
# SSH commands to check JTI/gRPC streaming telemetry capabilities on QFX5100
# Run these against your switches (10.10.10.55 and 10.10.10.56)

SWITCH_IP="${1:-10.10.10.55}"
SSH_KEY="/home/multiplex/.ssh/id_rsa"

echo "========================================================================"
echo "CHECKING TELEMETRY CAPABILITIES ON ${SWITCH_IP}"
echo "========================================================================"
echo ""

echo "1. Check Junos version and chassis (verify 21.4R3):"
ssh -i ${SSH_KEY} root@${SWITCH_IP} "cli -c 'show version | match \"Junos:|Model\"'"
echo ""

echo "2. Check if 'analytics' (JTI) is available in configuration hierarchy:"
ssh -i ${SSH_KEY} root@${SWITCH_IP} "cli -c 'show configuration services | display set | match analytics'"
echo ""

echo "3. Check if telemetry/analytics services are supported:"
ssh -i ${SSH_KEY} root@${SWITCH_IP} "cli -c 'show services analytics ?'"
echo ""

echo "4. Check for gRPC support:"
ssh -i ${SSH_KEY} root@${SWITCH_IP} "cli -c 'show configuration system services | display set | match extension-service'"
echo ""

echo "5. Check if Junos Telemetry Interface (JTI) daemon is available:"
ssh -i ${SSH_KEY} root@${SWITCH_IP} "ps aux | grep -E '(jvisiond|analyticsd|sensord)'"
echo ""

echo "6. Check available sensor paths (if analytics supported):"
ssh -i ${SSH_KEY} root@${SWITCH_IP} "cli -c 'request system storage cleanup dry-run'"
ssh -i ${SSH_KEY} root@${SWITCH_IP} "cli -c 'show agent sensors'"
echo ""

echo "7. Check firewall counter export capabilities:"
#ssh -i ${SSH_KEY} root@${SWITCH_IP} "cli -c 'show configuration firewall | display set | match \"then count\"'"
echo ""

echo "8. Check if OpenConfig/gNMI is available (alternative telemetry):"
ssh -i ${SSH_KEY} root@${SWITCH_IP} "cli -c 'show system processes | match gnmi'"
echo ""

echo "9. List available event-options (for scripting fallback):"
ssh -i ${SSH_KEY} root@${SWITCH_IP} "cli -c 'show configuration event-options | display set'"
echo ""

echo "10. Check if sFlow/sampling is available (packet-based alternative):"
ssh -i ${SSH_KEY} root@${SWITCH_IP} "cli -c 'show configuration protocols sflow | display set'"
echo ""

echo "========================================================================"
echo "ADDITIONAL CHECKS"
echo "========================================================================"
echo ""

echo "11. Check installed software packages (looking for telemetry packages):"
ssh -i ${SSH_KEY} root@${SWITCH_IP} "cli -c 'show system software packages'"
echo ""

echo "12. Check UDP port availability (default gRPC ports: 50051, 32767):"
ssh -i ${SSH_KEY} root@${SWITCH_IP} "netstat -an | grep -E ':(50051|32767|32768)'"
echo ""

echo "========================================================================"
echo "DONE - Review output above to determine telemetry capabilities"
echo "========================================================================"


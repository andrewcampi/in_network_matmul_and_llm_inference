# Hardware Limits

Below are the limitations discovered of these particular switches. If we pass these limits, the configs might apply, but our code will silently fail. 


- Less than 8 VLANs per trunk per switch
- Less than ~1,152 terms per filter TCAM limit per switch
-  3 VLAN filter limit per switch (confirmed in e103-e105):
  - Only 3 VLANs can have attached firewall filters simultaneously
  - Pattern: VLANs 0-2 work, 3+ fail (tested with ranges 200-211 AND 100-111)
  - NOT a VLAN number issue - it's a filter processing limit
- Python not supported on switch
- Switch uses csh, not bash
- VLAN Priority (PCP) not supported
- SCP does not work, we have to use SSH directly. Sending config files, then loading them on switch, is more efficient!


Notes:
The cleanup order matters:
❌ Incorrect: Delete filters → leaves orphaned VLAN references
✓ Correct: Delete VLAN attachments → Delete VLANs → Delete filters



HYPOTHESIS:
  Use 802.1p CoS priority to multiplex 8 neurons per MAC address
  Expected: 8× TCAM efficiency (360 terms instead of 2880)
FINDINGS:
  ✗ Junos ethernet-switching firewall filters CANNOT match on CoS priority
  ✗ 'user-priority' field does not exist in filter match conditions
  ✗ CoS classification happens AFTER firewall filtering
IMPLICATION:
  To count packets by CoS, we'd still need 8 separate filter terms
  Result: NO TCAM savings (still 8 terms for 8 neurons)
CONCLUSION:
  CoS queue multiplexing does NOT provide TCAM efficiency gains
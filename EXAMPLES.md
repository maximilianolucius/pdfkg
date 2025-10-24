# Example Questions for PDF Knowledge Graph

## What Can You Ask?

The knowledge graph enables several types of questions:

### 1. Direct Factual Questions

These questions look for specific information in the manual:

```
What are the operative temperature limits?
What is the input voltage range?
What are the dimensions?
What is the maximum current?
What certifications does this product have?
What is the IP protection rating?
What materials is the housing made of?
```

**How it works:**
- Embeddings find chunks containing keywords like "temperature", "voltage", "dimensions"
- Returns exact specifications from the manual with page/section references

### 2. Product Comparison Questions

```
What are the different models that exist?
What is the difference between model X and model Y?
What accessories are available?
What optional features can be added?
```

**How it works:**
- Finds sections discussing models/variants
- May reference comparison tables or product family sections

### 3. Installation & Procedure Questions

```
How do you mount this device?
What tools are required for installation?
How do I connect the wiring?
What is the installation sequence?
How do I configure the device?
```

**How it works:**
- Retrieves procedural text from installation sections
- Graph traversal finds related figures (mounting diagrams, wiring schematics)
- Cross-references like "see Figure 5" are resolved

### 4. Troubleshooting Questions

```
What should I do if the device doesn't power on?
How do I reset the device to factory settings?
What do the LED indicators mean?
Why is the device not responding?
```

**How it works:**
- Finds troubleshooting/diagnostics sections
- May reference diagnostic tables or flowcharts

### 5. Reference Lookup Questions

```
What figures show the mounting procedure?
Which section discusses safety precautions?
Where can I find the wiring diagram?
What page has the electrical specifications?
Where is the warranty information?
```

**How it works:**
- Uses cross-reference edges in the graph
- Traverses `REFERS_TO` edges from paragraphs to figures/tables/sections
- Returns page numbers and section IDs

### 6. Regulatory & Compliance Questions

```
What safety standards does this comply with?
What are the environmental operating conditions?
Is this device RoHS compliant?
What EMC standards are met?
```

**How it works:**
- Finds certification/compliance sections
- Often references standards tables

## Example Session

```bash
$ python chatbot.py --out data/out --use-gemini

Question: What are the operative temperature limits?

================================================================================
Q: What are the operative temperature limits?
================================================================================

Based on the manual, the operative temperature range is -25°C to +60°C.
This is specified in Section 3.2 Technical Specifications on page 12.

--- Sources ---
[1] Section 3.2, Page 12 (score: 0.892)
[2] Section 8.1, Page 45 (score: 0.734)

Related Figures: figure:3:p12
```

```bash
Question: How do you mount this device?

================================================================================
Q: How do you mount this device?
================================================================================

The device can be mounted in three ways:

1. **Wall mounting**: Use the supplied bracket and M4 screws (see Figure 8)
2. **DIN rail mounting**: Snap the DIN rail adapter onto a 35mm DIN rail
3. **Panel mounting**: Use the panel cutout template in Appendix B

Detailed instructions are in Section 4.3 Installation on page 18.

--- Sources ---
[1] Section 4.3, Page 18 (score: 0.845)
[2] Section 4.4, Page 21 (score: 0.712)

Related Figures: figure:8:p19, figure:9:p20
Related Sections: 4.3, 4.4
```

## What Makes a Good Question?

✅ **Good questions:**
- Specific: "What is the IP rating?" (not "Tell me about protection")
- Use domain terminology: "operative temperature" vs "how hot can it get"
- Ask about documented facts, not opinions or predictions

❌ **Questions the system can't answer well:**
- Comparative questions requiring external knowledge: "Is this better than competitor X?"
- Calculations not in the manual: "How many can I chain together?"
- Future/hypothetical: "Will this work with next-gen devices?"
- Questions requiring visual interpretation: "What color is the LED?" (unless stated in text)

## Tips for Best Results

1. **Be specific**: Instead of "How does it work?", ask "How does the startup sequence work?"

2. **Use --verbose flag**: See which chunks/pages were used and what related content exists
   ```bash
   python chatbot.py --out data/out --verbose
   ```

3. **Try both modes**:
   - Without Gemini: Fast, shows raw chunks with page numbers
   - With Gemini: Synthesizes natural language answers, better for complex questions

4. **Follow up on cross-references**: If an answer mentions "see Figure 5", ask "What does Figure 5 show?"

5. **Check related content**: Use `--verbose` to see related figures/tables/sections for deeper exploration

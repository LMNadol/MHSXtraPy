from . import read as rd
import igraph as ig
import networkx as nx
import os

def plot(filename, layout='fr', labels=False, save=False, vsize=10, errors=False, show=True, hcs=False):
    nulls = rd.nulls(filename)
    con = rd.separators(filename, lines=False)
    prefile = rd.prefix(filename)

    g = ig.Graph()
    g.add_vertices(nulls.number[-1])
    for inull in range(nulls.number[-1]):
        if nulls[inull].sign != 0:
            if nulls[inull].sign == 1:
                col = 'red'
            elif nulls[inull].sign == -1:
                col = 'blue'
            g.vs[inull]['color'] = col
            if labels == True:
                g.vs[inull]['label'] = '{}'.format(inull+1)

    for inull, cnulls in enumerate(con):
        toremove = []
        for jnull in cnulls:
            if inull+1 in con[jnull-1]:
                g.add_edge(inull, jnull-1)
                con[jnull-1].remove(inull+1)
                toremove.append(jnull)
                # con[inull].remove(jnull)
            else:
                if errors == True:
                    g.add_edge(inull, jnull-1, color=g.vs[inull]['color'])
                else:
                    g.add_edge(inull, jnull-1)
                if cnulls.count(jnull) > 1:
                    cnulls.remove(jnull)
        for iremove in toremove:
            con[inull].remove(iremove)

    # plot all seps using colours and no reduction
    # for inull, cnulls in enumerate(con):
    #     print(inull+1, cnulls)
    #     for jnull in cnulls:
    #         g.add_edge(inull, jnull-1, color=g.vs[inull]['color'])

    if os.path.isfile('output/'+prefile+'-hcs-connectivity.dat') and hcs:
        conhcs = rd.separators(filename, lines=False, hcs=True)
        for ihcs, connect in enumerate(conhcs):
            if len(connect) > 0:
                g.add_vertex(1)
                g.vs[ihcs+nulls.number[-1]]['color'] = 'green'
                for inull in connect:
                    g.add_edge(ihcs+nulls.number[-1], inull-1)

    if show or save:
        layout = g.layout(layout)
        if save == False:
            ig.plot(g, layout=layout, bbox=(1000,1000), margin=50, vertex_size=vsize)
        else:
            ig.plot(g, 'figures/'+prefile+'-graph.png', layout=layout, bbox=(1000,1000), margin=50, vertex_size=vsize)
    
    return g

def newplot(filename):
    print('Filename is {}'.format(filename))
    nulls = rd.nulls(filename)
    con = rd.separators(filename, lines=False)
    prefile = rd.prefix(filename)

    g = nx.Graph()
    g.add_nodes_from([i for i in range(1, nulls.number[-1]+1)])

    blackedges = []
    blueedges = []
    rededges = []
    for inull, cnulls in enumerate(con, start=1):
        toremove = []
        for jnull in cnulls:
            g.add_edge(inull, jnull)
            if inull in con[jnull-1]:
                blackedges.append((inull, jnull))
                con[jnull-1].remove(inull)
                toremove.append(jnull)
            else:
                if nulls[nulls.number == inull].sign == 1:
                    rededges.append((inull, jnull))
                else:
                    blueedges.append((inull, jnull))
        for iremove in toremove:
            con[inull-1].remove(iremove)

    if os.path.isfile('output/'+prefile+'-hcs-connectivity.dat') and hcs:
        conhcs = rd.separators(filename, lines=False, hcs=True)
        for ihcs, connect in enumerate(conhcs, start=1):
            if len(connect) > 0:
                g.add_node(nulls.number[-1] + ihcs)
                for inull in connect:
                    g.add_edge(ihcs + nulls.number[-1], inull)
                    blackedges.append((ihcs + nulls.number[-1], inull))

    pos = nx.fruchterman_reingold_layout(g)

    nx.draw(g, pos)


  
